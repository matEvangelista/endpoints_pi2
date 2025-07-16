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
from fastapi import FastAPI, Depends, HTTPException, status, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase, Driver, Session
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import jwt as pyjwt
from datetime import datetime, timedelta, timezone
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import os
from fastapi.staticfiles import StaticFiles

# --- 2. CONFIGURAÇÃO DO BANCO DE DADOS E MODELO DE ML ---
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_AUTH = ("neo4j", "senha123456789") # Senha que você forneceu

SECRET_KEY = "your-secret-key-here-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

security = HTTPBearer()

# Objetos que armazenarão o driver e o modelo
db_driver: Optional[Driver] = None
embedding_model: Optional[SentenceTransformer] = None

UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
    name: str = Field(..., example="Maria")
    surname: str = Field(..., example="Silva")
    email: str = Field(..., example="usuario@email.com")

class UsuarioCreate(UsuarioBase):
    password: str = Field(..., example="senhaSuperForte123")

class UsuarioUpdate(BaseModel):
    name: Optional[str] = None
    surname: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None

class UsuarioResponse(UsuarioBase):
    user_id: int = Field(..., example=1)

# Schemas de Coleção
class ColecaoCreate(BaseModel):
    nome: str = Field(..., example="Meus Favoritos")

class ColecaoResponse(ColecaoCreate):
    id: str
    user_id: int
    livros: List[Dict[str, Any]] = Field(default_factory=list)

class ColecoesResponse(ColecaoCreate):
    id: str
    livros: List[Dict[str, Any]] = Field(default_factory=list)

class ColecaoUpdate(BaseModel): # Schema adicionado
    nome: Optional[str] = None
    emoji: Optional[str] = None

# Schemas de Comentários
class ComentarioBase(BaseModel):
    texto: str = Field(..., example="Um livro transformador!")

class ComentarioCreate(ComentarioBase):
    pass

class ComentarioUpdate(ComentarioBase):
    pass

class ComentarioResponse(ComentarioBase):
    id: str
    user_id: int
    user_name: str
    livro_id: int
    data_criacao: datetime

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

# Schemas de autenticação
class LoginRequest(BaseModel):
    email: str = Field(..., example="email@exemplo.com")
    senha: str = Field(..., example="senhaSuperForte123")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int


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
    # Busca o maior user_id atual e incrementa
    result = session.run("OPTIONAL MATCH (u:Usuario) RETURN coalesce(max(u.id), 0) AS max_id")
    max_id = result.single()["max_id"]
    user_id = max_id + 1
    senha_hash = bcrypt.hashpw(usuario.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    query = """
        CREATE (u:Usuario {id: $user_id, name: $name, surname: $surname, email: $email, password: $senha_hash})
        RETURN u.id as user_id, u.name as name, u.surname as surname, u.email as email
    """
    params = {"user_id": user_id, "name": usuario.name, "surname": usuario.surname, "email": usuario.email, "senha_hash": senha_hash}
    result = session.run(query, params)
    return result.single().data()

def crud_ler_usuario(session: Session, user_id: int) -> Optional[Dict[str, Any]]:
    result = session.run("MATCH (u:Usuario {id: $user_id}) RETURN u.id as user_id, u.name as name, u.surname as surname, u.email as email", user_id=user_id)
    return result.single().data() if result.peek() else None

def crud_atualizar_usuario(session: Session, user_id: int, usuario_update: UsuarioUpdate) -> Optional[Dict[str, Any]]:
    update_data = usuario_update.dict(exclude_unset=True)
    if not update_data: return crud_ler_usuario(session, user_id)
    if "password" in update_data:
        update_data["password"] = bcrypt.hashpw(update_data.pop("password").encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    query = "MATCH (u:Usuario {id: $user_id}) SET u += $updates RETURN u.id as user_id, u.name as name, u.surname as surname, u.email as email"
    result = session.run(query, user_id=user_id, updates=update_data)
    return result.single().data() if result.peek() else None

def crud_apagar_usuario(session: Session, user_id: int) -> bool:
    result = session.run("MATCH (u:Usuario {id: $user_id}) DETACH DELETE u", user_id=user_id)
    return result.consume().counters.nodes_deleted > 0

# --- CRUD de Coleção ---
def crud_ler_colecoes_por_usuario(session: Session, user_id: int) -> List[Dict[str, Any]]:
    query = """
    MATCH (u:Usuario {id: $user_id})-[:CRIOU]->(c:Colecao)
    OPTIONAL MATCH (c)-[:CONTEM]->(l:Livro)
    RETURN c.id as id, c.nome as nome, u.id as user_id, collect({id: l.id, titulo: l.titulo}) as livros
    """
    result = session.run(query, user_id=user_id)
    return [record.data() for record in result]

def crud_criar_colecao(session: Session, user_id: int, colecao: ColecaoCreate) -> Dict[str, Any]:
    colecao_id = f"col_{user_id}_{re.sub(r'[^a-z0-9_]', '', colecao.nome.lower())}"
    query = "MATCH (u:Usuario {id: $user_id}) MERGE (u)-[:CRIOU]->(c:Colecao {id: $colecao_id}) ON CREATE SET c.nome = $nome RETURN c.id as id, c.nome as nome, u.id as user_id"
    result = session.run(query, user_id=user_id, colecao_id=colecao_id, nome=colecao.nome)
    return result.single().data()

def crud_pegar_colecoes(session: Session, user_id: int) -> List[Dict[str, Any]]:
    query = "MATCH (u:Usuario {id: $user_id})-[:CRIOU]->(c:Colecao) RETURN c.id AS id, c.nome AS nome"
    result = session.run(query, user_id=user_id)
    return result.data()


def crud_ler_colecao(session: Session, colecao_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    query = """
    MATCH (u:Usuario {id: $user_id})-[:CRIOU]->(c:Colecao {id: $id})
    OPTIONAL MATCH (c)-[:CONTEM]->(l:Livro)
    OPTIONAL MATCH (a:Autor)-[:ESCREVEU]->(l)
    WITH c, u, collect(CASE WHEN l IS NULL THEN {} ELSE {
        id: l.id, 
        titulo: coalesce(l.titulo, "Título Desconhecido"),
        url_img: coalesce(l.url_img, ""),
        descricao: coalesce(l.descricao, ""),
        ano_publicacao: l.ano_publicacao,
        autor: coalesce(a.nome, "Autor Desconhecido")
    } END) AS livros
    RETURN c.id AS id, c.nome AS nome, u.id AS user_id, livros
    """
    result = session.run(query, id=colecao_id, user_id=user_id)
    return result.single().data() if result.peek() else None

def crud_apagar_colecao(session: Session, colecao_id: str) -> bool:
    result = session.run("MATCH (c:Colecao {id: $id}) DETACH DELETE c", id=colecao_id)
    return result.consume().counters.nodes_deleted > 0

def crud_atualizar_colecao(session: Session, colecao_id: str, colecao_update: ColecaoUpdate) -> Optional[Dict[str, Any]]:
    update_data = colecao_update.dict(exclude_unset=True)
    if not update_data:
        return crud_ler_colecao(session, colecao_id)

    query = """
        MATCH (c:Colecao {id: $id})
        SET c += $updates
        RETURN c.id as id, c.nome as nome, c.emoji as emoji
    """
    result = session.run(query, id=colecao_id, updates=update_data)
    return result.single().data() if result.peek() else None

# --- CRUD de Relacionamentos ---
def crud_adicionar_livro_colecao(session: Session, colecao_id: str, livro_id: int) -> bool:
    query = "MATCH (c:Colecao {id: $colecao_id}) MATCH (l:Livro {id: $livro_id}) MERGE (c)-[:CONTEM]->(l)"
    return session.run(query, colecao_id=colecao_id, livro_id=livro_id).consume().counters.relationships_created > 0

def crud_remover_livro_colecao(session: Session, colecao_id: str, livro_id: int) -> bool:
    query = "MATCH (c:Colecao {id: $colecao_id})-[r:CONTEM]->(l:Livro {id: $livro_id}) DELETE r"
    return session.run(query, colecao_id=colecao_id, livro_id=livro_id).consume().counters.relationships_deleted > 0

def crud_gerenciar_interacao(session: Session, user_id: int, livro_id: int, interacao: InteracaoCreate) -> Optional[Dict[str, Any]]:
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

def crud_ler_interacao(session: Session, user_id: int, livro_id: int) -> Optional[Dict[str, Any]]:
    query = "MATCH (:Usuario {id: $user_id})-[r:INTERAGIU_COM]->(:Livro {id: $livro_id}) RETURN r.id as id, r.status as status, r.comentarios as comentarios"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    record = result.single()
    return record.data() if record else None

def crud_livros_por_status_interacao(session: Session, user_id: int, status: str) -> List[Dict[str, Any]]:
    """
    Busca todos os livros com os quais um usuário interagiu com um status específico (LIDO, LENDO, etc.).
    """
    query = """
    MATCH (u:Usuario {id: $user_id})-[r:INTERAGIU_COM {status: $status}]->(l:Livro)
    OPTIONAL MATCH (a:Autor)-[:ESCREVEU]->(l)
    OPTIONAL MATCH (l)-[:PERTENCE_A]->(c:Categoria)
    RETURN l.id as id,
           coalesce(l.titulo, "") AS titulo,
           l.ano_publicacao as ano_publicacao,
           coalesce(l.url_img, "") as url_img,
           coalesce(l.descricao, "") as descricao,
           coalesce(a.nome, "Desconhecido") as autor,
           collect(c.nome) as categorias
    """
    result = session.run(query, user_id=user_id, status=status)
    livros = []
    for record in result:
        data = record.data()
        # Garante que a url da imagem seja sempre uma string para evitar erros no Pydantic/Frontend
        if not isinstance(data.get('url_img'), str):
            data['url_img'] = ""
        livros.append(data)
    return livros

def crud_apagar_interacao(session: Session, user_id: int, livro_id: int) -> bool:
    query = "MATCH (:Usuario {id: $user_id})-[r:INTERAGIU_COM]->(:Livro {id: $livro_id}) DELETE r"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_deleted > 0

def crud_avaliar_livro(session: Session, user_id: int, livro_id: int, avaliacao: AvaliacaoCreate) -> bool:
    query = """
        MATCH (u:Usuario {id: $user_id}), (l:Livro {id: $livro_id}) MERGE (u)-[r:AVALIOU]->(l) SET r.nota = $nota
    """
    result = session.run(query, user_id=user_id, livro_id=livro_id, nota=avaliacao.nota)
    counters = result.consume().counters
    return counters.relationships_created > 0 or counters.properties_set > 0

def crud_ler_avaliacao(session: Session, user_id: int, livro_id: int) -> Optional[Dict[str, Any]]:
    query = "MATCH (:Usuario {id: $user_id})-[r:AVALIOU]->(:Livro {id: $livro_id}) RETURN r.nota as nota"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    record = result.single()
    return record.data() if record else None

def crud_apagar_avaliacao(session: Session, user_id: int, livro_id: int) -> bool:
    query = "MATCH (:Usuario {user_id: $user_id})-[r:AVALIOU]->(:Livro {id: $livro_id}) DELETE r"
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_deleted > 0

def crud_gerar_recomendacoes(session: Session, user_id: int) -> List[Dict[str, Any]]:
    # 1. Buscar todos os dados necessários do Neo4j de uma vez.
    user_id = int(user_id)
    
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
    
    result = session.run(query, user_id=user_id)
    data = result.single()
    
    if not data or not data['source_books']: 
        return []
    source_books = data['source_books']
    candidate_books = data['candidate_books']
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

def crud_associar_livro_usuario(session: Session, user_id: int, livro_id: int) -> bool:
    query = """
        MATCH (u:Usuario {id: $user_id}), (l:Livro {id: $livro_id})
        MERGE (u)-[:REGISTROU]->(l)
    """
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_created > 0

def crud_livros_registrados_por_usuario(session: Session, user_id: int) -> List[Dict[str, Any]]:
    query = '''
    MATCH (u:Usuario {id: $user_id})-[:REGISTROU]->(l:Livro)
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
    '''
    result = session.run(query, user_id=user_id)
    livros = []
    for record in result:
        data = record.data()
        if not isinstance(data.get('url_img'), str):
            data['url_img'] = ""
        livros.append(data)
    return livros

def crud_buscar_ultimos_livros_registrados_por_usuario(session: Session, user_id: int, limite: int) -> List[Dict[str, Any]]:
    """
    Busca os livros mais recentes registrados por um usuário, ordenados pela data de criação.
    """
    query = '''
    MATCH (u:Usuario {id: $user_id})-[:REGISTROU]->(l:Livro)
    OPTIONAL MATCH (a:Autor)-[:ESCREVEU]->(l)
    OPTIONAL MATCH (l)-[:PERTENCE_A]->(c:Categoria)
    // Retorna os dados necessários incluindo a data de criação para ordenação
    RETURN l.id as id,
           coalesce(l.titulo, "") AS titulo,
           l.ano_publicacao as ano_publicacao,
           coalesce(l.url_img, "") as url_img,
           coalesce(l.descricao, "") as descricao,
           l.descr_embedding as descr_embedding,
           coalesce(a.nome, "Desconhecido") as autor,
           collect(c.nome) as categorias,
           l.data_criacao as data_criacao
    // Ordena pela data de criação em ordem decrescente (mais novo primeiro)
    ORDER BY data_criacao DESC
    // Limita o número de resultados
    LIMIT $limite
    '''
    result = session.run(query, user_id=user_id, limite=limite)
    livros = []
    for record in result:
        data = record.data()
        # Remove a chave de data_criacao pois não faz parte do LivroResponse
        data.pop('data_criacao', None) 
        if not isinstance(data.get('url_img'), str):
            data['url_img'] = ""
        livros.append(data)
    return livros

def crud_favoritar_livro(session: Session, user_id: int, livro_id: int) -> bool:
    """Cria um relacionamento :FAVORITOU entre um usuário e um livro."""
    query = """
    MATCH (u:Usuario {id: $user_id}), (l:Livro {id: $livro_id})
    MERGE (u)-[:FAVORITOU]->(l)
    """
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_created > 0


def crud_desfavoritar_livro(session: Session, user_id: int, livro_id: int) -> bool:
    """Remove um relacionamento :FAVORITOU entre um usuário e um livro."""
    query = """
    MATCH (u:Usuario {id: $user_id})-[r:FAVORITOU]->(l:Livro {id: $livro_id})
    DELETE r
    """
    result = session.run(query, user_id=user_id, livro_id=livro_id)
    return result.consume().counters.relationships_deleted > 0

def crud_listar_livros_favoritos(session: Session, user_id: int) -> List[Dict[str, Any]]:
    """Lista todos os livros favoritados por um usuário."""
    query = '''
    MATCH (u:Usuario {id: $user_id})-[:FAVORITOU]->(l:Livro)
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
    '''
    result = session.run(query, user_id=user_id)
    livros = []
    for record in result:
        data = record.data()
        if not isinstance(data.get('url_img'), str):
            data['url_img'] = ""
        livros.append(data)
    return livros

# CRUD PARA COMENTÁRIOS
def crud_criar_comentario(session: Session, user_id: int, livro_id: int, comentario: ComentarioCreate) -> Dict[str, Any]:
    """Cria um novo comentário de um usuário para um livro."""
    query = """
    MATCH (u:Usuario {id: $user_id}), (l:Livro {id: $livro_id})
    CREATE (u)-[r:COMENTOU {
        id: randomUUID(),
        texto: $texto,
        data_criacao: datetime()
    }]->(l)
    RETURN r.id as id, r.texto as texto, r.data_criacao as data_criacao, u.id as user_id, u.name as user_name, l.id as livro_id
    """
    params = {"user_id": user_id, "livro_id": livro_id, "texto": comentario.texto}
    result = session.run(query, params)
    record = result.single()
    if not record:
        return None
    data = record.data()
    if data.get('data_criacao') and hasattr(data['data_criacao'], 'to_native'):
        data['data_criacao'] = data['data_criacao'].to_native()
    return data

def crud_ler_comentarios_de_um_livro(session: Session, livro_id: int) -> List[Dict[str, Any]]:
    """Lê todos os comentários de um livro específico."""
    query = """
    MATCH (u:Usuario)-[r:COMENTOU]->(l:Livro {id: $livro_id})
    RETURN r.id as id, r.texto as texto, r.data_criacao as data_criacao, u.id as user_id, u.name as user_name, l.id as livro_id
    ORDER BY r.data_criacao DESC
    """
    result = session.run(query, livro_id=livro_id)
    comentarios = []
    for record in result:
        data = record.data()
        if data.get('data_criacao') and hasattr(data['data_criacao'], 'to_native'):
            data['data_criacao'] = data['data_criacao'].to_native()
        comentarios.append(data)
    return comentarios

def crud_atualizar_comentario(session: Session, user_id: int, comentario_id: str, comentario_update: ComentarioUpdate) -> Optional[Dict[str, Any]]:
    """Atualiza um comentário, verificando se o usuário é o dono."""
    query = """
    MATCH (u:Usuario {id: $user_id})-[r:COMENTOU {id: $comentario_id}]->(l:Livro)
    SET r.texto = $texto
    RETURN r.id as id, r.texto as texto, r.data_criacao as data_criacao, u.id as user_id, u.name as user_name, l.id as livro_id
    """
    params = {"user_id": user_id, "comentario_id": comentario_id, "texto": comentario_update.texto}
    result = session.run(query, params)
    record = result.single()
    if not record:
        return None
    data = record.data()
    if data.get('data_criacao') and hasattr(data['data_criacao'], 'to_native'):
        data['data_criacao'] = data['data_criacao'].to_native()
    return data

def crud_apagar_comentario(session: Session, user_id: int, comentario_id: str) -> bool:
    """Apaga um comentário, verificando se o usuário é o dono."""
    query = "MATCH (u:Usuario {id: $user_id})-[r:COMENTOU {id: $comentario_id}]->() DELETE r"
    result = session.run(query, user_id=user_id, comentario_id=comentario_id)
    return result.consume().counters.relationships_deleted > 0

# Autenticaão
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def authenticate_user(session: Session, email: str, password: str) -> Optional[Dict[str, Any]]:
    # Find user by email
    result = session.run(
        "MATCH (u:Usuario {email: $email}) RETURN u.id as user_id, u.name as name, u.surname as surname, u.email as email, u.password as password",
        email=email
    )
    record = result.single()
    
    if not record:
        return None
    
    user_data = record.data()
    if not verify_password(password, user_data['password']):
        return None
    
    return user_data

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), session: Session = Depends(get_db_session)) -> Dict[str, Any]:
    try:
        payload = pyjwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = crud_ler_usuario(session, int(user_id))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    
    return user


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
    
@app.post("/usuarios/{user_id}/livros/", response_model=LivroResponse, status_code=status.HTTP_201_CREATED, tags=["Livros"])
def criar_livro_endpoint(user_id: int, livro: LivroCreate, session: Session = Depends(get_db_session), model: SentenceTransformer = Depends(get_embedding_model)):
    db_livro_info = crud_criar_livro(session, livro, model)
    if not db_livro_info:
        raise HTTPException(status_code=500, detail="Não foi possível criar o livro.")
    
    livro_id = db_livro_info['id']
    
    # Aqui associamos o livro recém-criado com o usuário que fez a requisição.
    crud_associar_livro_usuario(session, user_id, livro_id)
       
    return crud_ler_livro(session, livro_id)

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

# --- Rotas de Usuário ---
@app.post("/usuarios/", response_model=UsuarioResponse, status_code=status.HTTP_201_CREATED, tags=["Usuários"])
def criar_usuario_endpoint(usuario: UsuarioCreate, session: Session = Depends(get_db_session)):
    return crud_criar_usuario(session, usuario)

@app.get("/usuarios/{user_id}", response_model=UsuarioResponse, tags=["Usuários"])
def ler_usuario_endpoint(user_id: int, session: Session = Depends(get_db_session)):
    db_user = crud_ler_usuario(session, user_id)
    if not db_user: raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return db_user

@app.put("/usuarios/{user_id}", response_model=UsuarioResponse, tags=["Usuários"])
def atualizar_usuario_endpoint(user_id: int, usuario: UsuarioUpdate, session: Session = Depends(get_db_session)):
    db_user = crud_atualizar_usuario(session, user_id, usuario)
    if not db_user: raise HTTPException(status_code=404, detail="Usuário não encontrado")
    return db_user

@app.delete("/usuarios/{user_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Usuários"])
def apagar_usuario_endpoint(user_id: int, session: Session = Depends(get_db_session)):
    if not crud_apagar_usuario(session, user_id):
        raise HTTPException(status_code=404, detail="Usuário não encontrado")


@app.post("/usuarios/{user_id}/colecoes/", response_model=ColecaoResponse, tags=["Coleções"])
def criar_colecao_endpoint(user_id: int, colecao: ColecaoCreate, session: Session = Depends(get_db_session)):
    db_colecao = crud_criar_colecao(session, user_id, colecao)
    return crud_ler_colecao(session, db_colecao['id'], user_id)

@app.get("/usuarios/{user_id}/colecoes/", response_model=List[ColecoesResponse], tags=["Coleções"])
def pegar_colecoes_endpoint(user_id: int, session: Session = Depends(get_db_session)):
    db_colecao = crud_pegar_colecoes(session, user_id)
    return [ColecoesResponse(**colecao) for colecao in db_colecao]

@app.get("/colecoes/{colecao_id}", response_model=ColecaoResponse, tags=["Coleções"])
def ler_colecao_endpoint(user_id: int, colecao_id: str, session: Session = Depends(get_db_session)):
    db_colecao = crud_ler_colecao(session, colecao_id, user_id)
    if not db_colecao: raise HTTPException(status_code=404, detail="Coleção não encontrada")
    return db_colecao

@app.delete("/colecoes/{colecao_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Coleções"])
def apagar_colecao_endpoint(colecao_id: str, session: Session = Depends(get_db_session)):
    if not crud_apagar_colecao(session, colecao_id):
        raise HTTPException(status_code=404, detail="Coleção não encontrada")
    
@app.put("/colecoes/{colecao_id}", response_model=ColecaoResponse, tags=["Coleções"])
def atualizar_colecao_endpoint(colecao_id: str, colecao: ColecaoUpdate, session: Session = Depends(get_db_session)):
    db_colecao = crud_atualizar_colecao(session, colecao_id, colecao)
    if not db_colecao:
        raise HTTPException(status_code=404, detail="Coleção não encontrada para atualizar")
    result = session.run("MATCH (u:Usuario)-[:CRIOU]->(c:Colecao {id: $id}) RETURN u.id as user_id", id=colecao_id)
    owner = result.single()
    if not owner:
        raise HTTPException(status_code=404, detail="Dono da coleção não encontrado")
    
    return crud_ler_colecao(session, colecao_id, owner['user_id'])

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
def gerenciar_interacao_endpoint(user_id: int, livro_id: int, interacao: InteracaoCreate, session: Session = Depends(get_db_session)):
    resultado = crud_gerenciar_interacao(session, user_id, livro_id, interacao)
    if resultado is None: raise HTTPException(status_code=404, detail="Usuário ou Livro não encontrado")
    return resultado

@app.get("/usuarios/{user_id}/livros/{livro_id}/interacao", response_model=InteracaoResponse, tags=["Interações"])
def ler_interacao_endpoint(user_id: int, livro_id: int, session: Session = Depends(get_db_session)):
    resultado = crud_ler_interacao(session, user_id, livro_id)
    if resultado is None: raise HTTPException(status_code=404, detail="Interação não encontrada")
    return resultado

@app.delete("/usuarios/{user_id}/livros/{livro_id}/interacao", status_code=status.HTTP_204_NO_CONTENT, tags=["Interações"])
def apagar_interacao_endpoint(user_id: int, livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_apagar_interacao(session, user_id, livro_id):
        raise HTTPException(status_code=404, detail="Interação não encontrada")
    
@app.get("/usuarios/{user_id}/livros/status/{status}", response_model=List[LivroResponse], tags=["Interações"])
def listar_livros_por_status_endpoint(user_id: int, status: str, session: Session = Depends(get_db_session)):
    """
    Lista todos os livros de um usuário filtrados por um status de interação.
    O status deve ser 'LIDO' ou 'LENDO'.
    """
    valid_statuses = ["LIDO", "LENDO", "NOVO"]
    if status.upper() not in valid_statuses:
        raise HTTPException(status_code=400, detail="Status inválido. Use 'LIDO' ou 'LENDO'.")
    
    livros = crud_livros_por_status_interacao(session, user_id, status.upper())
    return livros

@app.post("/usuarios/{user_id}/livros/{livro_id}/favoritar", status_code=status.HTTP_201_CREATED, tags=["Favoritos"])
def favoritar_livro_endpoint(user_id: int, livro_id: int, session: Session = Depends(get_db_session)):
    """Marca um livro como favorito para um usuário."""
    crud_favoritar_livro(session, user_id, livro_id)

    return {"message": "Livro marcado como favorito com sucesso."}

@app.delete("/usuarios/{user_id}/livros/{livro_id}/favoritar", status_code=status.HTTP_204_NO_CONTENT, tags=["Favoritos"])
def desfavoritar_livro_endpoint(user_id: int, livro_id: int, session: Session = Depends(get_db_session)):
    """Remove um livro dos favoritos de um usuário."""
    if not crud_desfavoritar_livro(session, user_id, livro_id):
        raise HTTPException(status_code=404, detail="Relação de favorito não encontrada para ser removida.")

@app.get("/usuarios/{user_id}/favoritos", response_model=List[LivroResponse], tags=["Favoritos"])
def listar_favoritos_endpoint(user_id: int, session: Session = Depends(get_db_session)):
    """Lista todos os livros favoritados por um usuário."""
    livros_favoritos = crud_listar_livros_favoritos(session, user_id)
    return livros_favoritos

@app.post("/usuarios/{user_id}/livros/{livro_id}/avaliar", status_code=status.HTTP_204_NO_CONTENT, tags=["Interações"])
def avaliar_livro_endpoint(user_id: int, livro_id: int, avaliacao: AvaliacaoCreate, session: Session = Depends(get_db_session)):
    if not crud_avaliar_livro(session, user_id, livro_id, avaliacao):
        raise HTTPException(status_code=404, detail="Usuário ou livro não encontrado para avaliar")

@app.get("/usuarios/{user_id}/livros/{livro_id}/avaliacao", response_model=AvaliacaoResponse, tags=["Interações"])
def ler_avaliacao_endpoint(user_id: int, livro_id: int, session: Session = Depends(get_db_session)):
    resultado = crud_ler_avaliacao(session, user_id, livro_id)
    if resultado is None: raise HTTPException(status_code=404, detail="Avaliação não encontrada")
    return resultado

@app.delete("/usuarios/{user_id}/livros/{livro_id}/avaliacao", status_code=status.HTTP_204_NO_CONTENT, tags=["Interações"])
def apagar_avaliacao_endpoint(user_id: int, livro_id: int, session: Session = Depends(get_db_session)):
    if not crud_apagar_avaliacao(session, user_id, livro_id):
        raise HTTPException(status_code=404, detail="Avaliação não encontrada")

@app.get("/usuarios/{user_id}/recomendacoes", response_model=RecomendacoesResponse, tags=["Recomendações"])
def gerar_recomendacoes_endpoint(user_id: int, session: Session = Depends(get_db_session)):
    recomendacoes = crud_gerar_recomendacoes(session, user_id)
    return {"recomendacoes": recomendacoes}

@app.get("/usuarios/{user_id}/livros/registrados", response_model=List[LivroResponse], tags=["Livros"])
def listar_livros_registrados_endpoint(user_id: int, session: Session = Depends(get_db_session)):
    livros = crud_livros_registrados_por_usuario(session, user_id)
    return livros

@app.get("/usuarios/{user_id}/livros/recentes", response_model=List[LivroResponse], tags=["Livros"])
def listar_ultimos_livros_registrados_endpoint(
    user_id: int, 
    limite: int = Query(5, ge=1, le=20, description="Número de livros recentes a serem retornados."),
    session: Session = Depends(get_db_session)
):
    """
    Obtém uma lista dos livros mais recentemente cadastrados por um usuário específico.
    """
    livros = crud_buscar_ultimos_livros_registrados_por_usuario(session, user_id, limite)
    if not livros:
        raise HTTPException(status_code=404, detail="Nenhum livro registrado encontrado para este usuário.")
    return livros

# ROTAS PARA COMENTÁRIOS
@app.post("/usuarios/{user_id}/livros/{livro_id}/comentarios", response_model=ComentarioResponse, status_code=status.HTTP_201_CREATED, tags=["Comentários"])
def criar_comentario_endpoint(user_id: int, livro_id: int, comentario: ComentarioCreate, session: Session = Depends(get_db_session)):
    """Cria um novo comentário de um usuário para um livro."""

    novo_comentario = crud_criar_comentario(session, user_id, livro_id, comentario)
    if not novo_comentario:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Não foi possível adicionar o comentário. Verifique se o usuário e o livro existem."
        )
    return novo_comentario

@app.get("/livros/{livro_id}/comentarios", response_model=List[ComentarioResponse], tags=["Comentários"])
def ler_comentarios_endpoint(livro_id: int, session: Session = Depends(get_db_session)):
    """Lê todos os comentários de um livro específico."""
    return crud_ler_comentarios_de_um_livro(session, livro_id)

@app.put("/usuarios/{user_id}/comentarios/{comentario_id}", response_model=ComentarioResponse, tags=["Comentários"])
def atualizar_comentario_endpoint(user_id: int, comentario_id: str, comentario: ComentarioUpdate, session: Session = Depends(get_db_session)):
    """Atualiza um comentário existente. O usuário deve ser o autor do comentário."""

    comentario_atualizado = crud_atualizar_comentario(session, user_id, comentario_id, comentario)
    if not comentario_atualizado:
        raise HTTPException(status_code=404, detail="Comentário não encontrado ou você não tem permissão para editá-lo.")
    return comentario_atualizado

@app.delete("/usuarios/{user_id}/comentarios/{comentario_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Comentários"])
def apagar_comentario_endpoint(user_id: int, comentario_id: str, session: Session = Depends(get_db_session)):
    """Apaga um comentário. O usuário deve ser o autor do comentário."""

    if not crud_apagar_comentario(session, user_id, comentario_id):
        raise HTTPException(status_code=404, detail="Comentário não encontrado ou você não tem permissão para apagá-lo.")



@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
def registar_usuario_endpoint(usuario: UsuarioCreate, session: Session = Depends(get_db_session)):
    # Verifica se o usuário já existe
    result = session.run("MATCH (u:Usuario {email: $email}) RETURN u.id", email=usuario.email)
    if result.single():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Registra o usuário
    user_data = crud_criar_usuario(session, usuario)
    
    # Cria um token de acesso para o usuário
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_data["user_id"])}, expires_delta=access_token_expires
    )
    user_id = str(user_data["user_id"])
    
    return TokenResponse(access_token=str(access_token), user_id=user_id)

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
def login_endpoint(login_data: LoginRequest, session: Session = Depends(get_db_session)):
    user = authenticate_user(session, login_data.email, login_data.senha)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user["user_id"])}, expires_delta=access_token_expires
    )
    
    return TokenResponse(access_token=access_token, user_id=str(user["user_id"]))

@app.get("/auth/me", response_model=UsuarioResponse, tags=["Authentication"])
def get_current_user_endpoint(current_user: Dict[str, Any] = Depends(get_current_user)):
    return current_user

@app.post("/upload/imagem_livro/", tags=["Livros"])
def upload_imagem_livro(file: UploadFile = File(...)):
    """
    Recebe um arquivo de imagem, salva em disco e retorna a URL pública.
    """
    extensao = os.path.splitext(file.filename)[1]
    if extensao.lower() not in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
        return JSONResponse(status_code=400, content={"erro": "Formato de imagem não suportado."})
    nome_arquivo = f"livro_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}{extensao}"
    caminho_arquivo = os.path.join(UPLOAD_DIR, nome_arquivo)
    with open(caminho_arquivo, "wb") as buffer:
        buffer.write(file.file.read())
    url_publica = f"/static/{nome_arquivo}"
    return {"url_img": url_publica}

@app.delete("/upload/imagem_livro/", tags=["Livros"])
def delete_imagem_livro(url_img: str = Query(...)):
    """
    Deleta um arquivo de imagem do servidor.
    """
    try:
        # Extract filename from URL
        if url_img.startswith("/static/"):
            filename = url_img.replace("/static/", "")
        elif url_img.startswith("http://127.0.0.1:8000/static/"):
            filename = url_img.replace("http://127.0.0.1:8000/static/", "")
        else:
            return JSONResponse(status_code=400, content={"erro": "URL de imagem inválida."})
        
        caminho_arquivo = os.path.join(UPLOAD_DIR, filename)
        
        if os.path.exists(caminho_arquivo):
            os.remove(caminho_arquivo)
            return {"mensagem": "Imagem deletada com sucesso."}
        else:
            return JSONResponse(status_code=404, content={"erro": "Arquivo não encontrado."})
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": f"Erro ao deletar imagem: {str(e)}"})

from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")