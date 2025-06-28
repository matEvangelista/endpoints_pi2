# --- 1. CONFIGURAÇÃO E DEPENDÊNCIAS ---
#
# Para rodar este serviço, você precisará instalar as seguintes bibliotecas:
# pip install "fastapi[all]" uvicorn neo4j bcrypt sentence-transformers numpy
#
# Para executar, salve este arquivo como `main.py` e rode no terminal:
# uvicorn main:app --reload
#
# Acesse a documentação interativa da API em: http://127.0.0.1:8000/docs

import re
from typing import List, Optional, Dict, Any

import bcrypt
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase, Driver, Session
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- 2. CONFIGURAÇÃO DO BANCO DE DADOS E MODELO DE ML ---

NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "neo4j") # Senha que você forneceu

# Objetos que armazenarão o driver e o modelo
db_driver: Optional[Driver] = None
embedding_model: Optional[SentenceTransformer] = None

def get_db_session() -> Session:
    """
    Dependency Injection: Cria e fornece uma sessão do Neo4j para cada requisição.
    """
    if db_driver is None:
        raise HTTPException(status_code=503, detail="A conexão com o banco de dados não está disponível.")
    return db_driver.session(database="neo4j")

def get_embedding_model() -> SentenceTransformer:
    """
    Dependency Injection: Fornece a instância do modelo de embedding.
    """
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="O modelo de embedding não está disponível.")
    return embedding_model

async def lifespan(app: FastAPI):
    # Ao iniciar a aplicação
    global db_driver, embedding_model
    
    # Conecta ao Neo4j
    print("Conectando ao Neo4j...")
    db_driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
    try:
        db_driver.verify_connectivity()
        print("Conexão com Neo4j estabelecida com sucesso.")
    except Exception as e:
        print(f"Falha ao conectar com o Neo4j: {e}")
        db_driver = None
    
    # Carrega o modelo de embedding
    print("Carregando o modelo de Sentence Transformer...")
    try:
        embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("Modelo de embedding carregado com sucesso.")
    except Exception as e:
        print(f"Falha ao carregar o modelo de embedding: {e}")
        embedding_model = None
        
    yield
    
    # Ao desligar a aplicação
    if db_driver:
        print("Fechando a conexão com o Neo4j...")
        db_driver.close()
        print("Conexão fechada.")


# --- 3. SCHEMAS (Modelos Pydantic) ---

# Schemas de Autor
class AutorBase(BaseModel):
    nome: str = Field(..., example="George Orwell")

class AutorCreate(AutorBase):
    pass

class AutorUpdate(BaseModel):
    nome: Optional[str] = None

class AutorResponse(AutorBase):
    id: str = Field(..., example="george_orwell")
    livros: List[str] = Field([], example=["1984", "A Revolução dos Bichos"])

class AutorSearchResponse(BaseModel):
    resultados: List[AutorResponse]

# Schemas de Livro
class LivroBase(BaseModel):
    titulo: str = Field(..., example="1984")
    ano_publicacao: Optional[int] = Field(None, example=1949)
    url_img: Optional[str] = Field(None, example="http://exemplo.com/1984.jpg")
    descricao: Optional[str] = Field(None, example="Uma visão sombria de um futuro totalitário.")

class LivroCreate(LivroBase):
    autor_nome: str = Field(..., example="George Orwell")
    categorias: List[str] = Field(default_factory=list, example=["Distopia", "Ficção Científica"])

class LivroUpdate(BaseModel):
    titulo: Optional[str] = None
    ano_publicacao: Optional[int] = None
    url_img: Optional[str] = None
    descricao: Optional[str] = None
    categorias: Optional[List[str]] = None

class LivroResponse(LivroBase):
    id: int # ID numérico gerado pelo sistema
    autor: Optional[str] = Field(None, example="George Orwell")
    categorias: List[str] = Field(default_factory=list)
    descr_embedding: Optional[List[float]] = None

class LivroSearchResponse(BaseModel):
    resultados: List[LivroResponse]

# Schemas de Usuário
class UsuarioBase(BaseModel):
    nome: str = Field(..., example="Mateus")
    sobrenome: str = Field(..., example="Evangelista")
    email: str = Field(..., example="mateus.e@exemplo.com")

class UsuarioCreate(UsuarioBase):
    senha: str = Field(..., example="senhaSuperForte123")

class UsuarioUpdate(BaseModel):
    nome: Optional[str] = None
    sobrenome: Optional[str] = None
    email: Optional[str] = None
    senha: Optional[str] = None

class UsuarioResponse(UsuarioBase):
    id: str = Field(..., example="mateus_evangelista")

# Schemas de Coleção
class ColecaoCreate(BaseModel):
    nome: str = Field(..., example="Meus Favoritos")

class ColecaoResponse(ColecaoCreate):
    id: str
    user_id: str
    livros: List[Dict[str, Any]] = Field(default_factory=list)

# Schemas de Relacionamentos e Recomendações
class InteracaoCreate(BaseModel):
    status: str = Field(..., example="LENDO", pattern="^(LIDO|LENDO|NOVO)$")
    comentario: Optional[str] = Field(None, example="Este livro é incrível!")

class AvaliacaoCreate(BaseModel):
    nota: int = Field(..., ge=1, le=5, example=5)

class AvaliacaoResponse(BaseModel):
    nota: int

class InteracaoResponse(BaseModel):
    id: str
    status: str
    comentarios: List[str] = Field(default_factory=list)

class Recomendacao(BaseModel):
    id: int
    titulo: str
    similaridade: float
    motivo: str

class RecomendacoesResponse(BaseModel):
    recomendacoes: List[Recomendacao]


# --- 4. FUNÇÕES CRUD (Lógica de Banco de Dados) ---

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calcula a similaridade de cosseno entre dois vetores usando numpy."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

# --- CRUD de Autor ---
def crud_criar_autor(session: Session, autor: AutorCreate) -> Optional[Dict[str, Any]]:
    autor_id = re.sub(r'[^a-z0-9_]', '', re.sub(r'\s+', '_', autor.nome.lower()))
    result = session.run("MERGE (a:Autor {id: $id}) ON CREATE SET a.nome = $nome RETURN a.id as id, a.nome as nome", id=autor_id, nome=autor.nome)
    return result.single().data()

def crud_ler_autor(session: Session, autor_id: str) -> Optional[Dict[str, Any]]:
    query = "MATCH (a:Autor {id: $id}) OPTIONAL MATCH (a)-[:ESCREVEU]->(l:Livro) RETURN a.id AS id, coalesce(a.nome, '') AS nome, collect(l.titulo) AS livros"
    result = session.run(query, id=autor_id)
    record = result.single()
    return record.data() if record and record['id'] else None

def crud_buscar_autores_por_nome(session: Session, nome: str) -> List[Dict[str, Any]]:
    query = "MATCH (a:Autor) WHERE toLower(a.nome) CONTAINS toLower($nome) OPTIONAL MATCH (a)-[:ESCREVEU]->(l:Livro) RETURN a.id AS id, coalesce(a.nome, '') AS nome, collect(l.titulo) AS livros"
    result = session.run(query, nome=nome)
    return [record.data() for record in result]

def crud_atualizar_autor(session: Session, autor_id: str, autor_update: AutorUpdate) -> Optional[Dict[str, Any]]:
    update_data = autor_update.dict(exclude_unset=True)
    if not update_data: return crud_ler_autor(session, autor_id)
    query = "MATCH (a:Autor {id: $id}) SET a += $updates RETURN a.id as id, a.nome as nome"
    result = session.run(query, id=autor_id, updates=update_data)
    return result.single().data() if result.peek() else None

def crud_apagar_autor(session: Session, autor_id: str) -> bool:
    result = session.run("MATCH (a:Autor {id: $id}) DETACH DELETE a", id=autor_id)
    return result.consume().counters.nodes_deleted > 0

# --- CRUD de Livro ---
def crud_criar_livro(session: Session, livro: LivroCreate, model: SentenceTransformer) -> Optional[Dict[str, Any]]:
    autor_id = re.sub(r'[^a-z0-9_]', '', re.sub(r'\s+', '_', livro.autor_nome.lower()))
    
    descricao_limpa = livro.descricao if livro.descricao and livro.descricao != "None" else ""
    embedding = model.encode(descricao_limpa).tolist() if descricao_limpa else []
    
    query = """
        OPTIONAL MATCH (n:Livro) WITH coalesce(max(n.id), 0) AS max_id
        CREATE (l:Livro {
            id: max_id + 1,
            titulo: $titulo,
            ano_publicacao: $ano_publicacao,
            url_img: $url_img,
            descricao: $descricao,
            descr_embedding: $embedding,
            data_criacao: datetime()
        })
        WITH l
        MERGE (a:Autor {id: $autor_id}) ON CREATE SET a.nome = $autor_nome
        MERGE (a)-[:ESCREVEU]->(l)
        WITH l
        UNWIND $categorias as nome_cat
        MERGE (c:Categoria {id: toLower(replace(nome_cat, ' ', '_'))}) ON CREATE SET c.nome = nome_cat
        MERGE (l)-[:PERTENCE_A]->(c)
        RETURN l.id as id
        """
    params = {
        "titulo": livro.titulo,
        "ano_publicacao": livro.ano_publicacao,
        "url_img": livro.url_img,
        "descricao": descricao_limpa,
        "embedding": embedding,
        "autor_id": autor_id,
        "autor_nome": livro.autor_nome,
        "categorias": livro.categorias
    }
    result = session.run(query, params)
    return result.single()

def crud_ler_livro(session: Session, livro_id: int) -> Optional[Dict[str, Any]]:
    query = """
    MATCH (l:Livro {id: $id})
    OPTIONAL MATCH (a:Autor)-[:ESCREVEU]->(l)
    OPTIONAL MATCH (l)-[:PERTENCE_A]->(c:Categoria)
    RETURN l.id as id, 
           coalesce(l.titulo, "") AS titulo, 
           l.ano_publicacao as ano_publicacao, 
           coalesce(l.url_img, "") as url_img, 
           coalesce(l.descricao, "") as descricao,
           l.descr_embedding as descr_embedding, 
           coalesce(a.nome, "Desconhecido") as autor, 
           collect(c.nome) as categorias
    """
    result = session.run(query, id=livro_id)
    record = result.single()
    if not record:
        return None
    
    data = record.data()
    # Sanitize data to prevent validation errors with NaN values
    if not isinstance(data.get('url_img'), str):
        data['url_img'] = ""
    return data

def crud_buscar_livros_por_titulo(session: Session, titulo: str) -> List[Dict[str, Any]]:
    query = """
    MATCH (l:Livro) WHERE toLower(l.titulo) CONTAINS toLower($titulo) AND l.descricao <> "None"
    OPTIONAL MATCH (a:Autor)-[:ESCREVEU]->(l)
    OPTIONAL MATCH (l)-[:PERTENCE_A]->(c:Categoria)
    RETURN l.id as id, 
           coalesce(l.titulo, "") AS titulo, 
           l.ano_publicacao as ano_publicacao, 
           coalesce(l.url_img, "") as url_img, 
           coalesce(l.descricao, "") as descricao,
           l.descr_embedding as descr_embedding, 
           coalesce(a.nome, "Desconhecido") as autor, 
           collect(c.nome) as categorias
    """
    result = session.run(query, titulo=titulo)
    livros = []
    for record in result:
        data = record.data()
        # Sanitize data to prevent validation errors with NaN values
        if not isinstance(data.get('url_img'), str):
            data['url_img'] = ""
        livros.append(data)
    return livros

def crud_atualizar_livro(session: Session, livro_id: int, livro_update: LivroUpdate, model: SentenceTransformer) -> bool:
    update_data = livro_update.dict(exclude_unset=True)
    novas_categorias = update_data.pop('categorias', None)
    
    if 'descricao' in update_data:
        descricao_atualizada = update_data['descricao']
        if not descricao_atualizada or descricao_atualizada == "None":
            update_data['descricao'] = ""
            update_data['descr_embedding'] = []
        else:
            update_data['descr_embedding'] = model.encode(descricao_atualizada).tolist()

    properties_set = False
    if update_data:
        query_props = "MATCH (l:Livro {id: $id}) SET l += $updates"
        result_props = session.run(query_props, id=livro_id, updates=update_data)
        if result_props.consume().counters.properties_set > 0: properties_set = True

    if novas_categorias is not None:
        query_cats = """
            MATCH (l:Livro {id: $id})
            OPTIONAL MATCH (l)-[r:PERTENCE_A]->() DELETE r
            WITH l
            UNWIND $novas_categorias AS nome_cat
            MERGE (c:Categoria {id: toLower(replace(nome_cat, ' ', '_'))}) ON CREATE SET c.nome = nome_cat
            MERGE (l)-[:PERTENCE_A]->(c)
        """
        result_cats = session.run(query_cats, id=livro_id, novas_categorias=novas_categorias)
        if result_cats.consume().counters.relationships_created > 0: properties_set = True
    return properties_set

def crud_apagar_livro(session: Session, livro_id: int) -> bool:
    result = session.run("MATCH (l:Livro {id: $id}) DETACH DELETE l", id=livro_id)
    return result.consume().counters.nodes_deleted > 0

def crud_buscar_todos_livros(session: Session) -> List[Dict[str, Any]]:
    query = """
    MATCH (l:Livro)
    OPTIONAL MATCH (a:Autor)-[:ESCREVEU]->(l)
    OPTIONAL MATCH (l)-[:PERTENCE_A]->(c:Categoria)
    RETURN l.id as id, 
           coalesce(l.titulo, "") AS titulo, 
           l.ano_publicacao as ano_publicacao, 
           coalesce(l.url_img, "") as url_img, 
           coalesce(l.descricao, "") as descricao,
           l.descr_embedding as descr_embedding, 
           coalesce(a.nome, "Desconhecido") as autor, 
           collect(c.nome) as categorias
    """
    result = session.run(query)
    livros = []
    for record in result:
        data = record.data()
        # Sanitize data to prevent validation errors with NaN values
        if not isinstance(data.get('url_img'), str):
            data['url_img'] = ""
        livros.append(data)
    return livros

# --- CRUD de Usuário ---
def crud_criar_usuario(session: Session, usuario: UsuarioCreate) -> Dict[str, Any]:
    user_id = re.sub(r'[^a-z0-9_]', '', f"{usuario.nome.lower()}_{usuario.sobrenome.lower()}")
    senha_hash = bcrypt.hashpw(usuario.senha.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    query = "MERGE (u:Usuario {id: $id}) ON CREATE SET u.nome = $nome, u.sobrenome = $sobrenome, u.email = $email, u.senha_hash = $senha_hash RETURN u.id as id, u.nome as nome, u.sobrenome as sobrenome, u.email as email"
    params = usuario.dict(); params.update({"id": user_id, "senha_hash": senha_hash})
    result = session.run(query, params)
    return result.single().data()

def crud_ler_usuario(session: Session, user_id: str) -> Optional[Dict[str, Any]]:
    result = session.run("MATCH (u:Usuario {id: $id}) RETURN u.id as id, u.nome as nome, u.sobrenome as sobrenome, u.email as email", id=user_id)
    return result.single().data() if result.peek() else None

def crud_atualizar_usuario(session: Session, user_id: str, usuario_update: UsuarioUpdate) -> Optional[Dict[str, Any]]:
    update_data = usuario_update.dict(exclude_unset=True)
    if not update_data: return crud_ler_usuario(session, user_id)
    if "senha" in update_data:
        update_data["senha_hash"] = bcrypt.hashpw(update_data.pop("senha").encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    query = "MATCH (u:Usuario {id: $id}) SET u += $updates RETURN u.id as id, u.nome as nome, u.sobrenome as sobrenome, u.email as email"
    result = session.run(query, id=user_id, updates=update_data)
    return result.single().data() if result.peek() else None

def crud_apagar_usuario(session: Session, user_id: str) -> bool:
    result = session.run("MATCH (u:Usuario {id: $id}) DETACH DELETE u", id=user_id)
    return result.consume().counters.nodes_deleted > 0

# --- CRUD de Coleção ---
def crud_criar_colecao(session: Session, user_id: str, colecao: ColecaoCreate) -> Dict[str, Any]:
    colecao_id = f"col_{user_id}_{re.sub(r'[^a-z0-9_]', '', colecao.nome.lower())}"
    query = "MATCH (u:Usuario {id: $user_id}) MERGE (u)-[:CRIOU]->(c:Colecao {id: $colecao_id}) ON CREATE SET c.nome = $nome RETURN c.id as id, c.nome as nome, u.id as user_id"
    result = session.run(query, user_id=user_id, colecao_id=colecao_id, nome=colecao.nome)
    return result.single().data()

def crud_ler_colecao(session: Session, colecao_id: str) -> Optional[Dict[str, Any]]:
    query = """
    MATCH (c:Colecao {id: $id})<-[:CRIOU]-(u:Usuario)
    OPTIONAL MATCH (c)-[:CONTEM]->(l:Livro)
    WITH c, u, CASE WHEN l IS NULL THEN [] ELSE collect({id: l.id, titulo: coalesce(l.titulo, "Título Desconhecido")}) END as livros
    RETURN c.id as id, c.nome as nome, u.id as user_id, livros
    """
    result = session.run(query, id=colecao_id)
    return result.single().data() if result.peek() else None

def crud_apagar_colecao(session: Session, colecao_id: str) -> bool:
    result = session.run("MATCH (c:Colecao {id: $id}) DETACH DELETE c", id=colecao_id)
    return result.consume().counters.nodes_deleted > 0

# --- CRUD de Relacionamentos ---
def crud_adicionar_livro_colecao(session: Session, colecao_id: str, livro_id: int) -> bool:
    query = "MATCH (c:Colecao {id: $colecao_id}) MATCH (l:Livro {id: $livro_id}) MERGE (c)-[:CONTEM]->(l)"
    return session.run(query, colecao_id=colecao_id, livro_id=livro_id).consume().counters.relationships_created > 0

def crud_remover_livro_colecao(session: Session, colecao_id: str, livro_id: int) -> bool:
    query = "MATCH (c:Colecao {id: $colecao_id})-[r:CONTEM]->(l:Livro {id: $livro_id}) DELETE r"
    return session.run(query, colecao_id=colecao_id, livro_id=livro_id).consume().counters.relationships_deleted > 0

def crud_gerenciar_interacao(session: Session, user_id: str, livro_id: int, interacao: InteracaoCreate) -> Optional[Dict[str, Any]]:
    query = """
        MATCH (u:Usuario {id: $user_id}) MATCH (l:Livro {id: $livro_id})
        MERGE (u)-[r:INTERAGIU_COM]->(l)
        ON CREATE SET r.id=randomUUID(), r.status=$status, r.comentarios = CASE WHEN $novo_comentario IS NOT NULL THEN [$novo_comentario] ELSE [] END
        ON MATCH SET r.status=$status, r.comentarios = CASE WHEN $novo_comentario IS NOT NULL THEN r.comentarios + [$novo_comentario] ELSE r.comentarios END
        RETURN r.id as id, r.status as status, r.comentarios as comentarios
    """
    params = {"user_id": user_id, "livro_id": livro_id, "status": interacao.status, "novo_comentario": interacao.comentario}
    result = session.run(query, params)
    record = result.single()
    return record.data() if record else None

def crud_ler_interacao(session: Session, user_id: str, livro_id: int) -> Optional[Dict[str, Any]]:
    query = "MATCH (:Usuario {id: $user_id})-[r:INTERAGIU_COM]->(:Livro {id: $livro_id}) RETURN r.id as id, r.status as status, r.comentarios as comentarios"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    record = result.single()
    return record.data() if record else None

def crud_apagar_interacao(session: Session, user_id: str, livro_id: int) -> bool:
    query = "MATCH (:Usuario {id: $user_id})-[r:INTERAGIU_COM]->(:Livro {id: $livro_id}) DELETE r"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_deleted > 0

def crud_avaliar_livro(session: Session, user_id: str, livro_id: int, avaliacao: AvaliacaoCreate) -> bool:
    query = "MATCH (u:Usuario {id: $user_id}) MATCH (l:Livro {id: $livro_id}) MERGE (u)-[r:AVALIOU]->(l) SET r.nota = $nota"
    result = session.run(query, user_id=user_id, livro_id=livro_id, nota=avaliacao.nota)
    return result.consume().counters.properties_set > 0

def crud_ler_avaliacao(session: Session, user_id: str, livro_id: int) -> Optional[Dict[str, Any]]:
    query = "MATCH (:Usuario {id: $user_id})-[r:AVALIOU]->(:Livro {id: $livro_id}) RETURN r.nota as nota"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    record = result.single()
    return record.data() if record else None

def crud_apagar_avaliacao(session: Session, user_id: str, livro_id: int) -> bool:
    query = "MATCH (:Usuario {id: $user_id})-[r:AVALIOU]->(:Livro {id: $livro_id}) DELETE r"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_deleted > 0

def crud_gerar_recomendacoes(session: Session, user_id: str) -> List[Dict[str, Any]]:
    # 1. Buscar todos os dados necessários do Neo4j de uma vez.
    query = """
    MATCH (u:Usuario {id: $user_id})
    WITH u, [(u)-[:INTERAGIU_COM|:AVALIOU]->(b) | b.id] AS interacted_ids
    MATCH (u)-[a:AVALIOU]->(source_book:Livro)
    WHERE a.nota IN [4, 5] AND source_book.descr_embedding IS NOT NULL AND source_book.descricao <> "None"
    WITH u, interacted_ids, collect({id: source_book.id, titulo: source_book.titulo, embedding: source_book.descr_embedding, nota: a.nota}) AS source_books
    MATCH (candidate_book:Livro)
    WHERE candidate_book.descr_embedding IS NOT NULL AND NOT candidate_book.id IN interacted_ids AND candidate_book.descricao <> "None"
    WITH source_books, collect({id: candidate_book.id, titulo: candidate_book.titulo, embedding: candidate_book.descr_embedding}) AS candidate_books
    RETURN source_books, candidate_books
    """
    data = session.run(query, user_id=user_id).single()

    if not data or not data['source_books']: return []
    source_books = data['source_books']; candidate_books = data['candidate_books']
    all_recommendations = []

    for source_book in source_books:
        source_embedding = source_book.get('embedding')
        if not source_embedding: continue
        similarities = []
        for candidate_book in candidate_books:
            if source_book['id'] == candidate_book['id']: continue
            candidate_embedding = candidate_book.get('embedding')
            if not candidate_embedding: continue
            similarity = _cosine_similarity(source_embedding, candidate_embedding)
            similarities.append({"id": candidate_book['id'], "titulo": candidate_book['titulo'], "similaridade": similarity, "motivo": f"Similar a '{source_book['titulo']}' (nota {source_book['nota']})"})
        similarities.sort(key=lambda x: x['similaridade'], reverse=True)
        limit = 5 if source_book['nota'] == 5 else 1
        all_recommendations.extend(similarities[:limit])

    final_recs_dict = {}
    for rec in all_recommendations:
        book_id = rec['id']
        if book_id not in final_recs_dict or rec['similaridade'] > final_recs_dict[book_id]['similaridade']:
            final_recs_dict[book_id] = rec
    final_list = sorted(final_recs_dict.values(), key=lambda x: x['similaridade'], reverse=True)
    return final_list


# --- 5. ENDPOINTS DA API (Rotas) ---
app = FastAPI(title="API da Biblioteca", description="Um microsserviço para gerenciar livros, usuários e suas interações.", version="1.0.0", lifespan=lifespan)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/health", tags=["Health"])
def health_check():
    """Endpoint para verificar o status dos serviços."""
    status = {
        "status": "ok",
        "services": {
            "neo4j": "available" if db_driver is not None else "unavailable",
            "embedding_model": "available" if embedding_model is not None else "unavailable"
        }
    }
    
    if db_driver is None and embedding_model is None:
        status["status"] = "degraded"
        status["message"] = "Nenhum serviço crítico está disponível"
    elif db_driver is None:
        status["status"] = "degraded"
        status["message"] = "Neo4j não está disponível"
    elif embedding_model is None:
        status["status"] = "degraded"
        status["message"] = "Modelo de embedding não está disponível"
    
    return status

# --- Rotas de Autor ---
@app.get("/autores/buscar", response_model=AutorSearchResponse, tags=["Autores"])
def buscar_autores_endpoint(q: str = Query(..., min_length=3), session: Session = Depends(get_db_session)):
    autores = crud_buscar_autores_por_nome(session, q)
    return {"resultados": autores}

@app.post("/autores/", response_model=AutorResponse, status_code=status.HTTP_201_CREATED, tags=["Autores"])
def criar_autor_endpoint(autor: AutorCreate, session: Session = Depends(get_db_session)):
    db_autor = crud_criar_autor(session, autor)
    return crud_ler_autor(session, db_autor['id'])

@app.get("/autores/{autor_id}", response_model=AutorResponse, tags=["Autores"])
def ler_autor_endpoint(autor_id: str, session: Session = Depends(get_db_session)):
    db_autor = crud_ler_autor(session, autor_id)
    if not db_autor: raise HTTPException(status_code=404, detail="Autor não encontrado")
    return db_autor

@app.put("/autores/{autor_id}", response_model=AutorResponse, tags=["Autores"])
def atualizar_autor_endpoint(autor_id: str, autor: AutorUpdate, session: Session = Depends(get_db_session)):
    db_autor = crud_atualizar_autor(session, autor_id, autor)
    if not db_autor: raise HTTPException(status_code=404, detail="Autor não encontrado")
    return crud_ler_autor(session, autor_id)

@app.delete("/autores/{autor_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Autores"])
def apagar_autor_endpoint(autor_id: str, session: Session = Depends(get_db_session)):
    if not crud_apagar_autor(session, autor_id):
        raise HTTPException(status_code=404, detail="Autor não encontrado")
    return None

# --- Rotas de Livro ---
@app.get("/livros/buscar", response_model=LivroSearchResponse, tags=["Livros"])
def buscar_livros_endpoint(q: str = Query(..., min_length=3), session: Session = Depends(get_db_session)):
    livros = crud_buscar_livros_por_titulo(session, q)
    return {"resultados": livros}
    
@app.post("/livros/", response_model=LivroResponse, status_code=status.HTTP_201_CREATED, tags=["Livros"])
def criar_livro_endpoint(livro: LivroCreate, session: Session = Depends(get_db_session), model: SentenceTransformer = Depends(get_embedding_model)):
    db_livro_info = crud_criar_livro(session, livro, model)
    if not db_livro_info: raise HTTPException(status_code=500, detail="Não foi possível criar o livro.")
    return crud_ler_livro(session, db_livro_info['id'])

@app.get("/livros/{livro_id}", response_model=LivroResponse, tags=["Livros"])
def ler_livro_endpoint(livro_id: int, session: Session = Depends(get_db_session)):
    db_livro = crud_ler_livro(session, livro_id)
    if not db_livro: 
        raise HTTPException(status_code=404, detail="Livro não encontrado")
    if db_livro.get('descricao') == "None":
        raise HTTPException(status_code=404, detail="Livro não encontrado ou não disponível.")
    return db_livro

@app.put("/livros/{livro_id}", response_model=LivroResponse, tags=["Livros"])
def atualizar_livro_endpoint(livro_id: int, livro: LivroUpdate, session: Session = Depends(get_db_session), model: SentenceTransformer = Depends(get_embedding_model)):
    if not crud_atualizar_livro(session, livro_id, livro, model):
        raise HTTPException(status_code=404, detail="Livro não encontrado ou nenhuma propriedade foi alterada")
    return crud_ler_livro(session, livro_id)

@app.delete("/livros/{livro_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Livros"])
def apagar_livro_endpoint(livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_apagar_livro(session, livro_id):
        raise HTTPException(status_code=404, detail="Livro não encontrado")

@app.get("/livros/todos/", response_model=List[LivroResponse], tags=["Livros"])
def buscar_todos_livros_endpoint(session: Session = Depends(get_db_session)):
    livros = crud_buscar_todos_livros(session)
    return livros

# (Restante dos endpoints permanece o mesmo)
@app.post("/usuarios/", response_model=UsuarioResponse, status_code=status.HTTP_201_CREATED, tags=["Usuários"])
def criar_usuario_endpoint(usuario: UsuarioCreate, session: Session = Depends(get_db_session)):
    return crud_criar_usuario(session, usuario)

@app.get("/usuarios/{user_id}", response_model=UsuarioResponse, tags=["Usuários"])
def ler_usuario_endpoint(user_id: str, session: Session = Depends(get_db_session)):
    db_user = crud_ler_usuario(session, user_id)
    if not db_user: raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return db_user

@app.put("/usuarios/{user_id}", response_model=UsuarioResponse, tags=["Usuários"])
def atualizar_usuario_endpoint(user_id: str, usuario: UsuarioUpdate, session: Session = Depends(get_db_session)):
    db_user = crud_atualizar_usuario(session, user_id, usuario)
    if not db_user: raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return db_user

@app.delete("/usuarios/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Usuários"])
def apagar_usuario_endpoint(user_id: str, session: Session = Depends(get_db_session)):
    if not crud_apagar_usuario(session, user_id):
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

@app.post("/usuarios/{user_id}/colecoes/", response_model=ColecaoResponse, tags=["Coleções"])
def criar_colecao_endpoint(user_id: str, colecao: ColecaoCreate, session: Session = Depends(get_db_session)):
    db_colecao = crud_criar_colecao(session, user_id, colecao)
    return crud_ler_colecao(session, db_colecao['id'])

@app.get("/colecoes/{colecao_id}", response_model=ColecaoResponse, tags=["Coleções"])
def ler_colecao_endpoint(colecao_id: str, session: Session = Depends(get_db_session)):
    db_colecao = crud_ler_colecao(session, colecao_id)
    if not db_colecao: raise HTTPException(status_code=404, detail="Coleção não encontrada")
    return db_colecao

@app.delete("/colecoes/{colecao_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Coleções"])
def apagar_colecao_endpoint(colecao_id: str, session: Session = Depends(get_db_session)):
    if not crud_apagar_colecao(session, colecao_id):
        raise HTTPException(status_code=404, detail="Coleção não encontrada")

@app.post("/colecoes/{colecao_id}/livros/{livro_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Coleções"])
def adicionar_livro_a_colecao_endpoint(colecao_id: str, livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_adicionar_livro_colecao(session, colecao_id, livro_id):
        raise HTTPException(status_code=404, detail="Coleção ou livro não encontrado")

@app.delete("/colecoes/{colecao_id}/livros/{livro_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Coleções"])
def remover_livro_da_colecao_endpoint(colecao_id: str, livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_remover_livro_colecao(session, colecao_id, livro_id):
        raise HTTPException(status_code=404, detail="Relação Coleção-Livro não encontrada")

# --- Rotas de Interação e Recomendação ---
@app.post("/usuarios/{user_id}/livros/{livro_id}/interacao", response_model=InteracaoResponse, tags=["Interações"])
def gerenciar_interacao_endpoint(user_id: str, livro_id: int, interacao: InteracaoCreate, session: Session = Depends(get_db_session)):
    resultado = crud_gerenciar_interacao(session, user_id, livro_id, interacao)
    if resultado is None: raise HTTPException(status_code=404, detail="Usuário ou Livro não encontrado")
    return resultado

@app.get("/usuarios/{user_id}/livros/{livro_id}/interacao", response_model=InteracaoResponse, tags=["Interações"])
def ler_interacao_endpoint(user_id: str, livro_id: int, session: Session = Depends(get_db_session)):
    resultado = crud_ler_interacao(session, user_id, livro_id)
    if resultado is None: raise HTTPException(status_code=404, detail="Interação não encontrada")
    return resultado

@app.delete("/usuarios/{user_id}/livros/{livro_id}/interacao", status_code=status.HTTP_204_NO_CONTENT, tags=["Interações"])
def apagar_interacao_endpoint(user_id: str, livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_apagar_interacao(session, user_id, livro_id):
        raise HTTPException(status_code=404, detail="Interação não encontrada")

@app.post("/usuarios/{user_id}/livros/{livro_id}/avaliar", status_code=status.HTTP_204_NO_CONTENT, tags=["Interações"])
def avaliar_livro_endpoint(user_id: str, livro_id: int, avaliacao: AvaliacaoCreate, session: Session = Depends(get_db_session)):
    if not crud_avaliar_livro(session, user_id, livro_id, avaliacao):
        raise HTTPException(status_code=404, detail="Usuário ou livro não encontrado para avaliar")

@app.get("/usuarios/{user_id}/livros/{livro_id}/avaliacao", response_model=AvaliacaoResponse, tags=["Interações"])
def ler_avaliacao_endpoint(user_id: str, livro_id: int, session: Session = Depends(get_db_session)):
    resultado = crud_ler_avaliacao(session, user_id, livro_id)
    if resultado is None: raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    return resultado

@app.delete("/usuarios/{user_id}/livros/{livro_id}/avaliacao", status_code=status.HTTP_204_NO_CONTENT, tags=["Interações"])
def apagar_avaliacao_endpoint(user_id: str, livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_apagar_avaliacao(session, user_id, livro_id):
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")

@app.get("/usuarios/{user_id}/recomendacoes", response_model=RecomendacoesResponse, tags=["Recomendações"])
def gerar_recomendacoes_endpoint(user_id: str, session: Session = Depends(get_db_session)):
    recomendacoes = crud_gerar_recomendacoes(session, user_id)
    return {"recomendacoes": recomendacoes}
