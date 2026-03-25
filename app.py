import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Pesquisa Jurisprudencial",
    page_icon="⚖️",
    layout="wide"
)

# ==========================================
# Autenticação e Gestão de Estado
# ==========================================
def check_password():
    """Retorna True se o usuário estiver logado com a senha correta."""
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        _, col_main, _ = st.columns([1.5, 7.0, 1.5])
        with col_main:
            st.title("🔒 Acesso Restrito")
            password = st.text_input("Senha do Gabinete", type="password")
            
            if st.button("Entrar", type="primary"):
                # st.secrets gerencia de forma segura as variáveis no Streamlit Cloud
                if password == st.secrets.get("SENHA_GABINETE", ""):
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Senha incorreta.")
        return False
    return True

# ==========================================
# Caching e Vetorização (O Core de Performance)
# ==========================================
# Isolamos o carregamento na RAM para que ele aconteça apenas UMA vez para todos
@st.cache_resource(show_spinner="Carregando modelo de IA...")
def load_model():
    """Carrega o modelo na RAM global da aplicação."""
    return SentenceTransformer("intfloat/multilingual-e5-small")

# ==========================================
# Aplicação Principal (UI)
# ==========================================
def main():
    # Bloqueio de Segurança
    if not check_password():
        return

    # Restringindo a UI a 70% da tela usando colunas
    _, col_main, _ = st.columns([1.5, 7.0, 1.5])
    
    with col_main:
        st.title("⚖️ Pesquisa Semântica do Gabinete")
        st.markdown("Busca semântica de jurisprudência.")

        # Conexão com Pinecone
        try:
            pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
            index_name = st.secrets.get("PINECONE_INDEX_NAME", "default-index")
            index = pc.Index(index_name)
        except Exception as e:
            st.error("Falha ao inicializar banco de dados no Pinecone.")
            st.exception(e)
            return

        # Formulário de Pesquisa
        query_text = st.text_input("O que você deseja buscar?", placeholder="Ex: taxa de lixo progressiva")

        if st.button("Pesquisar", type="primary") and query_text:
            model = load_model()
            
            with st.spinner("Vetorizando pesquisa e cruzando base de dados..."):
                
                # Regra de Sobrevivência: O Prefixo E5 Exato
                formatted_query = f"query: {query_text}"
                query_vector = model.encode(formatted_query).tolist()
                
                # Busca no Banco Vetorial
                try:
                    resultados = index.query(
                        vector=query_vector,
                        top_k=100, # Busca profunda para garantir a listagem de todos resultados com > 50%
                        include_metadata=True
                    )
                except Exception as e:
                    st.error("Erro durante a consulta no banco de dados vetorial.")
                    st.exception(e)
                    return
                
                # Análise e Renderização Limpa
                matches = resultados.get("matches", [])
                
                # Deduplicação por arquivo de origem e Limiar de 50%
                seen_files = set()
                filtered_matches = []
                
                for match in matches:
                    score = match.get("score", 0.0)
                    score_pct = score * 100
                    metadata = match.get("metadata", {})
                    file_path = metadata.get("file_path", "Documento_Desconhecido")
                    
                    if score_pct >= 50.0 and file_path not in seen_files:
                        filtered_matches.append(match)
                        seen_files.add(file_path)
                
                # Separar resultados atualizados dos desatualizados
                resultados_atuais = []
                resultados_desatualizados = []
                
                for match in filtered_matches:
                    metadata = match.get("metadata", {})
                    status = metadata.get("status", "")
                    if "desatualizada" in str(status).strip().lower():
                        resultados_desatualizados.append(match)
                    else:
                        resultados_atuais.append(match)
                
                total = len(resultados_atuais) + len(resultados_desatualizados)
                st.success(f"{total} resultados relevantes encontrados.")
                st.divider()
                
                FONT = "font-family: Arial, sans-serif; font-size: 12pt; line-height: 1.6;"
                
                # === RESULTADOS PRINCIPAIS ===
                for match in resultados_atuais:
                    score = match.get("score", 0.0)
                    score_pct = score * 100
                    metadata = match.get("metadata", {})
                    file_path = metadata.get("file_path", "Documento")
                    
                    # Título com o Path
                    st.subheader(f"📄 {file_path}")
                    st.caption(f"**Relevância:** {score_pct:.1f}%")
                    
                    # Conteúdo da Nota
                    conteudo = metadata.get("conteudo", "")
                    if conteudo and conteudo != "Sem contexto.":
                        with st.expander("Ler Conteúdo da MOC / Jurisprudência", expanded=False):
                            html_text = f"<div style='font-family: Arial, sans-serif; font-size: 12pt; white-space: pre-wrap; line-height: 1.6;'>{conteudo}</div>"
                            st.markdown(html_text, unsafe_allow_html=True)
                            
                    # Metadado Crítico: Votos
                    votos = metadata.get("votos_aplicados", [])
                    
                    if not votos:
                        st.markdown(f"<div style='{FONT}'><b>Votos cadastrados:</b> Nenhum voto cadastrado</div>", unsafe_allow_html=True)
                    else:
                        votos_text = ", ".join([str(v) for v in votos]) if isinstance(votos, list) else str(votos)
                        st.markdown(f"<div style='{FONT}'><b>Votos cadastrados:</b> {votos_text}</div>", unsafe_allow_html=True)
                    
                    # Outros metadados
                    tags = metadata.get("tags", [])
                    if tags:
                        tags_text = ", ".join([str(t) for t in tags]) if isinstance(tags, list) else str(tags)
                        st.markdown(f"<div style='{FONT}'><b>Tags:</b> {tags_text}</div>", unsafe_allow_html=True)
                    
                    status = metadata.get("status")
                    if status:
                        st.markdown(f"<div style='{FONT}'><b>Status:</b> {status}</div>", unsafe_allow_html=True)

                    tipo = metadata.get("tipo")
                    if tipo:
                        st.markdown(f"<div style='{FONT}'><b>Tipo:</b> {tipo}</div>", unsafe_allow_html=True)
                        
                    st.divider()
                
                # === NOTAS DESATUALIZADAS (Seção compacta ao final) ===
                if resultados_desatualizados:
                    FONT_SM = "font-family: Arial, sans-serif; font-size: 10pt; line-height: 1.4; color: #888;"
                    st.markdown("---")
                    st.markdown(f"<div style='font-family: Arial, sans-serif; font-size: 13pt; color: #999; margin-bottom: 8px;'>⚠️ Notas Desatualizadas ({len(resultados_desatualizados)})</div>", unsafe_allow_html=True)
                    
                    for match in resultados_desatualizados:
                        score = match.get("score", 0.0)
                        score_pct = score * 100
                        metadata = match.get("metadata", {})
                        file_path = metadata.get("file_path", "Documento")
                        st.markdown(f"<div style='{FONT_SM}'>📄 <b>{file_path}</b> — Relevância: {score_pct:.1f}% — <i>Desatualizada</i></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
