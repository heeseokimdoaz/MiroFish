"""
LocalZep — Zep Cloud API 호환 로컬 구현체

Zep Cloud의 rate limit/episode limit 문제를 해결하기 위해
SQLite + OpenAI로 GraphRAG를 로컬에서 처리합니다.

지원 API (zep_cloud.client.Zep 호환):
- client.graph.create(graph_id, name=...)
- client.graph.delete(graph_id)
- client.graph.set_ontology(graph_ids, entities, edges)
- client.graph.add(graph_id, data, type, source_description)
- client.graph.add_batch(batches)
- client.graph.search(query, graph_id, limit, scope, reranker)
- client.graph.node.get(node_uuid)
- client.graph.node.get_by_graph_id(graph_id)
- client.graph.node.get_entity_edges(node_uuid)
- client.graph.edge.get_by_graph_id(graph_id)
- client.graph.episode.get(episode_uuid)
"""

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger

logger = get_logger('mirofish.local_zep')

# 싱글톤 스토어 (프로세스당 하나의 SQLite DB)
_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads', 'local_zep.db')
_DB_LOCK = threading.RLock()


def _ensure_db():
    """SQLite DB 및 테이블 초기화."""
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS graphs (
            graph_id TEXT PRIMARY KEY,
            name TEXT,
            ontology_json TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS episodes (
            uuid TEXT PRIMARY KEY,
            graph_id TEXT,
            data TEXT,
            type TEXT,
            source_description TEXT,
            processed INTEGER DEFAULT 0,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            uuid TEXT PRIMARY KEY,
            graph_id TEXT,
            name TEXT,
            labels TEXT,
            summary TEXT,
            attributes_json TEXT,
            embedding_json TEXT,
            created_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS edges (
            uuid TEXT PRIMARY KEY,
            graph_id TEXT,
            source_node_uuid TEXT,
            target_node_uuid TEXT,
            name TEXT,
            fact TEXT,
            episode_uuid TEXT,
            attributes_json TEXT,
            created_at REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_graph ON nodes(graph_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_graph ON edges(graph_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_node_uuid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_node_uuid)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_graph ON episodes(graph_id)")
    conn.commit()
    return conn


# ---------- 데이터 객체 (Zep SDK와 유사한 모양) ----------
class _Node:
    """Zep SDK 호환 노드 객체 (uuid와 uuid_ 둘 다 제공)."""
    def __init__(self, uuid, graph_id, name, labels=None, summary="",
                 attributes=None, created_at=0.0):
        self.uuid = uuid
        self.uuid_ = uuid  # Zep SDK alias
        self.graph_id = graph_id
        self.name = name
        self.labels = labels or []
        self.summary = summary
        self.attributes = attributes or {}
        self.created_at = created_at

    @classmethod
    def from_row(cls, row):
        return cls(
            uuid=row[0],
            graph_id=row[1],
            name=row[2],
            labels=json.loads(row[3]) if row[3] else [],
            summary=row[4] or "",
            attributes=json.loads(row[5]) if row[5] else {},
            created_at=row[6] or 0.0,
        )


class _Edge:
    """Zep SDK 호환 엣지 객체."""
    def __init__(self, uuid, graph_id, source_node_uuid, target_node_uuid,
                 name, fact="", episode_uuid=None, attributes=None, created_at=0.0):
        self.uuid = uuid
        self.uuid_ = uuid
        self.graph_id = graph_id
        self.source_node_uuid = source_node_uuid
        self.target_node_uuid = target_node_uuid
        self.name = name
        self.edge_name = name  # alias
        self.fact = fact
        self.episode_uuid = episode_uuid
        self.attributes = attributes or {}
        self.created_at = created_at

    @classmethod
    def from_row(cls, row):
        return cls(
            uuid=row[0],
            graph_id=row[1],
            source_node_uuid=row[2],
            target_node_uuid=row[3],
            name=row[4],
            fact=row[5] or "",
            episode_uuid=row[6],
            attributes=json.loads(row[7]) if row[7] else {},
            created_at=row[8] or 0.0,
        )


class _Episode:
    def __init__(self, uuid, graph_id, data, type, source_description="",
                 processed=False, created_at=0.0):
        self.uuid = uuid
        self.uuid_ = uuid
        self.graph_id = graph_id
        self.data = data
        self.type = type
        self.source_description = source_description
        self.processed = processed
        self.created_at = created_at


@dataclass
class _SearchResult:
    nodes: List[_Node] = field(default_factory=list)
    edges: List[_Edge] = field(default_factory=list)


# ---------- 임베딩 유틸 (간단한 캐싱) ----------
_EMBEDDING_CACHE: Dict[str, List[float]] = {}


def _embed(text: str, client: OpenAI) -> List[float]:
    """텍스트를 임베딩으로 변환 (캐싱)."""
    if not text:
        return []
    key = text[:500]
    if key in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[key]
    model = os.getenv('LLM_EMBEDDING_MODEL', 'text-embedding-3-small')
    try:
        res = client.embeddings.create(
            model=model,
            input=text[:8000],
        )
        emb = res.data[0].embedding
        if len(_EMBEDDING_CACHE) < 5000:
            _EMBEDDING_CACHE[key] = emb
        return emb
    except Exception as e:
        logger.warning(f"임베딩 생성 실패: {e}")
        return []


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------- LLM 엔티티 추출 ----------
def _extract_entities_and_edges(
    text: str,
    ontology: Dict[str, Any],
    client: OpenAI,
    model_name: str,
) -> Dict[str, Any]:
    """
    텍스트에서 온톨로지 기반 엔티티와 관계를 LLM으로 추출.
    """
    entity_types = ontology.get("entity_types", []) if ontology else []
    edge_types = ontology.get("edge_types", []) if ontology else []

    entity_type_names = [e.get("name", "") for e in entity_types if e.get("name")]
    edge_type_names = [e.get("name", "") for e in edge_types if e.get("name")]

    if not entity_type_names:
        entity_type_names = ["Person", "Organization"]

    prompt = f"""다음 텍스트에서 엔티티와 관계를 추출하세요.

## 허용 엔티티 유형
{', '.join(entity_type_names)}

## 허용 관계 유형
{', '.join(edge_type_names) if edge_type_names else '자유'}

## 텍스트
{text[:6000]}

## 출력 (JSON, 마크다운 금지)
{{
  "entities": [
    {{"name": "엔티티 이름", "type": "유형(위 목록에서)", "summary": "짧은 설명"}}
  ],
  "edges": [
    {{"source": "엔티티A", "target": "엔티티B", "name": "관계유형", "fact": "관계를 설명하는 문장"}}
  ]
}}

규칙:
- 엔티티 이름은 텍스트에 실제로 등장한 것만
- type은 허용 유형 중 하나로 매핑
- edges는 엔티티 간 실제 관계가 텍스트에 언급된 것만
- 중복 제거
- 한국어 유지"""

    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "당신은 지식 그래프 추출 전문가입니다. 순수 JSON만 반환하세요."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        content = res.choices[0].message.content
        data = json.loads(content)
        return {
            "entities": data.get("entities", []),
            "edges": data.get("edges", []),
        }
    except Exception as e:
        logger.warning(f"엔티티 추출 실패: {e}")
        return {"entities": [], "edges": []}


# ---------- 서브클라이언트들 (Zep SDK 구조 흉내) ----------
class _NodeClient:
    def __init__(self, parent):
        self._parent = parent

    def get(self, node_uuid: str = None, uuid_: str = None, uuid: str = None, **_):
        """Zep SDK 호환: uuid_, node_uuid, uuid 어떤 것이든 받음."""
        nid = node_uuid or uuid_ or uuid
        if not nid:
            return None
        with _DB_LOCK:
            conn = _ensure_db()
            row = conn.execute(
                "SELECT uuid, graph_id, name, labels, summary, attributes_json, created_at FROM nodes WHERE uuid=?",
                (nid,),
            ).fetchone()
            conn.close()
        return _Node.from_row(row) if row else None

    def get_by_graph_id(self, graph_id: str, limit: int = 10000, uuid_cursor: str = None, **_):
        with _DB_LOCK:
            conn = _ensure_db()
            if uuid_cursor:
                rows = conn.execute(
                    "SELECT uuid, graph_id, name, labels, summary, attributes_json, created_at FROM nodes WHERE graph_id=? AND uuid > ? ORDER BY uuid LIMIT ?",
                    (graph_id, uuid_cursor, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT uuid, graph_id, name, labels, summary, attributes_json, created_at FROM nodes WHERE graph_id=? ORDER BY uuid LIMIT ?",
                    (graph_id, limit),
                ).fetchall()
            conn.close()
        return [_Node.from_row(r) for r in rows]

    def get_entity_edges(self, node_uuid: str = None, uuid_: str = None, uuid: str = None, limit: int = 100, **_):
        nid = node_uuid or uuid_ or uuid
        if not nid:
            return []
        with _DB_LOCK:
            conn = _ensure_db()
            rows = conn.execute(
                "SELECT uuid, graph_id, source_node_uuid, target_node_uuid, name, fact, episode_uuid, attributes_json, created_at FROM edges WHERE source_node_uuid=? OR target_node_uuid=? LIMIT ?",
                (nid, nid, limit),
            ).fetchall()
            conn.close()
        return [_Edge.from_row(r) for r in rows]


class _EdgeClient:
    def __init__(self, parent):
        self._parent = parent

    def get_by_graph_id(self, graph_id: str, limit: int = 10000, uuid_cursor: str = None, **_):
        with _DB_LOCK:
            conn = _ensure_db()
            if uuid_cursor:
                rows = conn.execute(
                    "SELECT uuid, graph_id, source_node_uuid, target_node_uuid, name, fact, episode_uuid, attributes_json, created_at FROM edges WHERE graph_id=? AND uuid > ? ORDER BY uuid LIMIT ?",
                    (graph_id, uuid_cursor, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT uuid, graph_id, source_node_uuid, target_node_uuid, name, fact, episode_uuid, attributes_json, created_at FROM edges WHERE graph_id=? ORDER BY uuid LIMIT ?",
                    (graph_id, limit),
                ).fetchall()
            conn.close()
        return [_Edge.from_row(r) for r in rows]


class _EpisodeClient:
    def __init__(self, parent):
        self._parent = parent

    def get(self, episode_uuid: str = None, uuid_: str = None, **_):
        """Zep SDK 호환: uuid_ 또는 episode_uuid로 조회."""
        eid = uuid_ or episode_uuid
        if not eid:
            return None
        with _DB_LOCK:
            conn = _ensure_db()
            row = conn.execute(
                "SELECT uuid, graph_id, data, type, source_description, processed, created_at FROM episodes WHERE uuid=?",
                (eid,),
            ).fetchone()
            conn.close()
        if not row:
            return None
        return _Episode(
            uuid=row[0], graph_id=row[1], data=row[2], type=row[3],
            source_description=row[4] or "", processed=bool(row[5]),
            created_at=row[6] or 0.0,
        )


class _GraphClient:
    def __init__(self, parent):
        self._parent = parent
        self.node = _NodeClient(self)
        self.edge = _EdgeClient(self)
        self.episode = _EpisodeClient(self)

    def create(self, graph_id: str, name: str = "", description: str = "", **_):
        with _DB_LOCK:
            conn = _ensure_db()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO graphs (graph_id, name, ontology_json, created_at) VALUES (?, ?, ?, ?)",
                    (graph_id, name, "{}", time.time()),
                )
                conn.commit()
            finally:
                conn.close()
        logger.info(f"[LocalZep] 그래프 생성: {graph_id}")
        return {"graph_id": graph_id, "name": name}

    def delete(self, graph_id: str):
        with _DB_LOCK:
            conn = _ensure_db()
            try:
                conn.execute("DELETE FROM nodes WHERE graph_id=?", (graph_id,))
                conn.execute("DELETE FROM edges WHERE graph_id=?", (graph_id,))
                conn.execute("DELETE FROM episodes WHERE graph_id=?", (graph_id,))
                conn.execute("DELETE FROM graphs WHERE graph_id=?", (graph_id,))
                conn.commit()
            finally:
                conn.close()
        logger.info(f"[LocalZep] 그래프 삭제: {graph_id}")

    def set_ontology(self, graph_ids=None, entities=None, edges=None, **_):
        """온톨로지 저장. entities/edges는 Pydantic 모델 리스트 또는 dict 리스트."""
        ontology = {
            "entity_types": [],
            "edge_types": [],
        }
        if entities:
            for e in entities:
                if hasattr(e, 'model_dump'):
                    ontology["entity_types"].append(e.model_dump())
                elif isinstance(e, dict):
                    ontology["entity_types"].append(e)
                else:
                    ontology["entity_types"].append({"name": getattr(e, "__name__", str(e))})
        if edges:
            for e in edges:
                if hasattr(e, 'model_dump'):
                    ontology["edge_types"].append(e.model_dump())
                elif isinstance(e, dict):
                    ontology["edge_types"].append(e)
                else:
                    ontology["edge_types"].append({"name": getattr(e, "__name__", str(e))})

        target_ids = graph_ids or []
        if not isinstance(target_ids, (list, tuple)):
            target_ids = [target_ids]

        ontology_json = json.dumps(ontology, ensure_ascii=False)
        with _DB_LOCK:
            conn = _ensure_db()
            try:
                for gid in target_ids:
                    conn.execute(
                        "INSERT OR REPLACE INTO graphs (graph_id, name, ontology_json, created_at) VALUES (?, COALESCE((SELECT name FROM graphs WHERE graph_id=?), ''), ?, COALESCE((SELECT created_at FROM graphs WHERE graph_id=?), ?))",
                        (gid, gid, ontology_json, gid, time.time()),
                    )
                conn.commit()
            finally:
                conn.close()
        logger.info(f"[LocalZep] 온톨로지 설정: {target_ids}, entities={len(ontology['entity_types'])}, edges={len(ontology['edge_types'])}")

    def _get_ontology(self, graph_id: str) -> Dict[str, Any]:
        with _DB_LOCK:
            conn = _ensure_db()
            row = conn.execute(
                "SELECT ontology_json FROM graphs WHERE graph_id=?",
                (graph_id,),
            ).fetchone()
            conn.close()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return {}
        return {}

    def _process_episode(self, episode_uuid: str, graph_id: str, data: str):
        """에피소드에서 엔티티/관계 추출 및 저장."""
        ontology = self._get_ontology(graph_id)
        client = self._parent._openai_client
        model = self._parent._model_name

        extracted = _extract_entities_and_edges(data, ontology, client, model)
        entities = extracted.get("entities", [])
        edges_data = extracted.get("edges", [])

        name_to_uuid: Dict[str, str] = {}
        now = time.time()

        with _DB_LOCK:
            conn = _ensure_db()
            try:
                # 기존 노드 재사용: 같은 그래프 내 동일 이름이면 재사용
                for ent in entities:
                    ename = (ent.get("name") or "").strip()
                    if not ename:
                        continue
                    etype = ent.get("type") or "Entity"
                    esummary = ent.get("summary") or ""
                    existing = conn.execute(
                        "SELECT uuid FROM nodes WHERE graph_id=? AND name=? LIMIT 1",
                        (graph_id, ename),
                    ).fetchone()
                    if existing:
                        name_to_uuid[ename] = existing[0]
                        if esummary:
                            conn.execute(
                                "UPDATE nodes SET summary=? WHERE uuid=?",
                                (esummary, existing[0]),
                            )
                        continue
                    node_uuid = str(uuid.uuid4())
                    # 임베딩은 검색 시점에 생성하도록 비워둠 (성능)
                    conn.execute(
                        "INSERT INTO nodes (uuid, graph_id, name, labels, summary, attributes_json, embedding_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            node_uuid, graph_id, ename,
                            json.dumps([etype, "Entity"]),
                            esummary, "{}", "", now,
                        ),
                    )
                    name_to_uuid[ename] = node_uuid

                # 엣지 생성
                for edg in edges_data:
                    src = (edg.get("source") or "").strip()
                    tgt = (edg.get("target") or "").strip()
                    ename = edg.get("name") or "RELATED_TO"
                    fact = edg.get("fact") or ""
                    if not src or not tgt:
                        continue
                    # 엔티티가 엔티티 목록에 없으면 Person으로 기본 생성
                    for n in (src, tgt):
                        if n not in name_to_uuid:
                            existing = conn.execute(
                                "SELECT uuid FROM nodes WHERE graph_id=? AND name=? LIMIT 1",
                                (graph_id, n),
                            ).fetchone()
                            if existing:
                                name_to_uuid[n] = existing[0]
                            else:
                                nu = str(uuid.uuid4())
                                conn.execute(
                                    "INSERT INTO nodes (uuid, graph_id, name, labels, summary, attributes_json, embedding_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                    (nu, graph_id, n, json.dumps(["Entity"]), "", "{}", "", now),
                                )
                                name_to_uuid[n] = nu
                    src_uuid = name_to_uuid[src]
                    tgt_uuid = name_to_uuid[tgt]
                    edge_uuid = str(uuid.uuid4())
                    conn.execute(
                        "INSERT INTO edges (uuid, graph_id, source_node_uuid, target_node_uuid, name, fact, episode_uuid, attributes_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (edge_uuid, graph_id, src_uuid, tgt_uuid, ename, fact, episode_uuid, "{}", now),
                    )

                # 에피소드 처리 완료 표시
                conn.execute(
                    "UPDATE episodes SET processed=1 WHERE uuid=?",
                    (episode_uuid,),
                )
                conn.commit()
            finally:
                conn.close()

        logger.info(f"[LocalZep] 에피소드 처리 완료: entities={len(entities)}, edges={len(edges_data)}")

    def add(self, graph_id: str, data: str, type: str = "text",
            source_description: str = "", **_):
        """에피소드 추가 및 동기 처리."""
        episode_uuid = str(uuid.uuid4())
        with _DB_LOCK:
            conn = _ensure_db()
            try:
                conn.execute(
                    "INSERT INTO episodes (uuid, graph_id, data, type, source_description, processed, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                    (episode_uuid, graph_id, data, type, source_description, time.time()),
                )
                conn.commit()
            finally:
                conn.close()

        # 비동기 처리가 이상적이지만, 간단히 동기로 처리
        try:
            self._process_episode(episode_uuid, graph_id, data)
        except Exception as e:
            logger.warning(f"[LocalZep] 에피소드 처리 실패: {e}")

        return {"uuid": episode_uuid, "uuid_": episode_uuid, "processed": True}

    def add_batch(self, batches=None, graph_id=None, episodes=None, **_):
        """배치 에피소드 추가 — 병렬 LLM 추출로 속도 개선 (Zep SDK 호환)."""
        import concurrent.futures
        results = []

        # Zep SDK 스타일: graph_id + episodes=[EpisodeData]
        if episodes is not None and graph_id is not None:
            # 1) 모든 에피소드를 먼저 DB에 저장 (processed=0)
            ep_infos = []
            with _DB_LOCK:
                conn = _ensure_db()
                try:
                    for ep in episodes:
                        if hasattr(ep, 'model_dump'):
                            ep_dict = ep.model_dump()
                        elif isinstance(ep, dict):
                            ep_dict = ep
                        else:
                            ep_dict = {
                                "data": getattr(ep, 'data', ''),
                                "type": getattr(ep, 'type', 'text'),
                                "source_description": getattr(ep, 'source_description', ''),
                            }
                        eu = str(uuid.uuid4())
                        conn.execute(
                            "INSERT INTO episodes (uuid, graph_id, data, type, source_description, processed, created_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                            (eu, graph_id, ep_dict.get("data", ""), ep_dict.get("type", "text"),
                             ep_dict.get("source_description", ""), time.time()),
                        )
                        ep_infos.append((eu, ep_dict.get("data", "")))
                    conn.commit()
                finally:
                    conn.close()

            # 2) 병렬로 엔티티/관계 추출 (최대 10개 동시)
            def _process_one(ep_uuid, data):
                try:
                    self._process_episode(ep_uuid, graph_id, data)
                except Exception as e:
                    logger.warning(f"[LocalZep] 에피소드 처리 실패: {e}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(_process_one, eu, data) for eu, data in ep_infos]
                for _ in concurrent.futures.as_completed(futures):
                    pass

            # 3) 결과 래핑
            class _Ep:
                pass
            for eu, _ in ep_infos:
                wrapper = _Ep()
                wrapper.uuid = eu
                wrapper.uuid_ = eu
                wrapper.processed = True
                results.append(wrapper)
            return results

        # 구형: batches=[dict]
        if batches:
            for b in batches:
                if hasattr(b, 'model_dump'):
                    b = b.model_dump()
                elif not isinstance(b, dict):
                    continue
                results.append(self.add(
                    graph_id=b.get("graph_id"),
                    data=b.get("data", ""),
                    type=b.get("type", "text"),
                    source_description=b.get("source_description", ""),
                ))
        return results

    def search(self, query: str, graph_id: str = None, limit: int = 10,
               scope: str = "edges", reranker: str = "rrf", **_):
        """임베딩 기반 유사도 검색."""
        client = self._parent._openai_client
        q_emb = _embed(query, client) if query else []

        with _DB_LOCK:
            conn = _ensure_db()
            if scope == "nodes":
                rows = conn.execute(
                    "SELECT uuid, graph_id, name, labels, summary, attributes_json, created_at, embedding_json FROM nodes WHERE graph_id=?",
                    (graph_id,) if graph_id else ("",),
                ).fetchall() if graph_id else []
                items = []
                for r in rows:
                    node = _Node.from_row(r[:7])
                    text_for_emb = f"{node.name} {node.summary}"
                    emb_json = r[7]
                    if emb_json:
                        try:
                            emb = json.loads(emb_json)
                        except Exception:
                            emb = []
                    else:
                        emb = _embed(text_for_emb, client) if q_emb else []
                        if emb:
                            conn.execute(
                                "UPDATE nodes SET embedding_json=? WHERE uuid=?",
                                (json.dumps(emb), node.uuid),
                            )
                    score = _cosine(q_emb, emb) if q_emb and emb else 0.0
                    # 키워드 매칭 보너스
                    if query and (query.lower() in (node.name or "").lower() or query.lower() in (node.summary or "").lower()):
                        score += 0.3
                    items.append((score, node))
                conn.commit()
                conn.close()
                items.sort(key=lambda x: x[0], reverse=True)
                return _SearchResult(nodes=[n for _, n in items[:limit]], edges=[])
            else:
                # edges
                rows = conn.execute(
                    "SELECT uuid, graph_id, source_node_uuid, target_node_uuid, name, fact, episode_uuid, attributes_json, created_at FROM edges WHERE graph_id=?",
                    (graph_id,) if graph_id else ("",),
                ).fetchall() if graph_id else []
                items = []
                for r in rows:
                    edge = _Edge.from_row(r)
                    text_for_emb = edge.fact or edge.name
                    emb = _embed(text_for_emb, client) if q_emb else []
                    score = _cosine(q_emb, emb) if q_emb and emb else 0.0
                    if query and query.lower() in (edge.fact or "").lower():
                        score += 0.3
                    items.append((score, edge))
                conn.close()
                items.sort(key=lambda x: x[0], reverse=True)
                return _SearchResult(nodes=[], edges=[e for _, e in items[:limit]])


# ---------- 메인 LocalZep 클라이언트 ----------
class LocalZep:
    """Zep Cloud `Zep` 클래스 호환 대체 구현."""

    def __init__(self, api_key: str = None, **_):
        # api_key는 무시 (로컬이므로 불필요)
        self._openai_client = OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.LLM_BASE_URL,
        )
        self._model_name = Config.LLM_MODEL_NAME
        _ensure_db().close()
        self.graph = _GraphClient(self)


def get_client(api_key: str = None) -> Any:
    """
    Zep 클라이언트 팩토리. ZEP_API_KEY가 설정되어 있고 USE_LOCAL_ZEP이 false면
    실제 Zep Cloud를 사용하고, 그렇지 않으면 LocalZep을 반환.
    """
    use_local = os.getenv('USE_LOCAL_ZEP', 'true').lower() in ('true', '1', 'yes')
    if use_local:
        logger.info("[LocalZep] 로컬 Zep 모드로 동작 (무료/무제한)")
        return LocalZep(api_key=api_key)
    try:
        from zep_cloud.client import Zep as ZepCloud
        logger.info("[LocalZep] Zep Cloud 모드로 동작")
        return ZepCloud(api_key=api_key)
    except Exception:
        logger.warning("[LocalZep] Zep Cloud import 실패, 로컬 모드로 폴백")
        return LocalZep(api_key=api_key)
