import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ==========================================
# Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Uniformizador de Jurisprudência",
    page_icon="⚖️",
    layout="wide"
)

# Esconde TUDO do Streamlit — a UI inteira vem do HTML
st.markdown("""
<style>
    .block-container { padding: 0 !important; max-width: 100% !important; }
    header, footer, .stDeployButton, [data-testid="stToolbar"] { display: none !important; }
    iframe { border: none !important; }
    [data-testid="stAppViewBlockContainer"] { padding: 0 !important; }
    .stApp > div:first-child { padding: 0 !important; }
    section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Autenticação
# ==========================================
def check_password():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    if not st.session_state["logged_in"]:
        _, col_main, _ = st.columns([1.5, 7.0, 1.5])
        with col_main:
            st.title("🔒 Acesso Restrito")
            password = st.text_input("Senha do Gabinete", type="password")
            if st.button("Entrar", type="primary"):
                if password == st.secrets.get("SENHA_GABINETE", ""):
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("Senha incorreta.")
        return False
    return True

# ==========================================
# Modelo
# ==========================================
@st.cache_resource(show_spinner="Carregando modelo de IA...")
def load_model():
    return SentenceTransformer("intfloat/multilingual-e5-small")

# ==========================================
# Busca
# ==========================================
def do_search(query_text, index):
    model = load_model()
    formatted_query = f"query: {query_text}"
    query_vector = model.encode(formatted_query).tolist()

    resultados = index.query(
        vector=query_vector,
        top_k=100,
        include_metadata=True
    )

    matches = resultados.get("matches", [])
    seen_files = set()
    results_for_ui = []
    counter = 0

    for match in matches:
        score = match.get("score", 0.0)
        metadata = match.get("metadata", {})
        file_path = metadata.get("file_path", "Documento_Desconhecido")

        if score * 100 >= 30.0 and file_path not in seen_files:
            seen_files.add(file_path)
            counter += 1

            votos = metadata.get("votos_aplicados", [])
            if isinstance(votos, str):
                votos = [v.strip() for v in votos.split(",") if v.strip()]

            tags = metadata.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]

            status = metadata.get("status", "Atual")
            if "desatualizada" in str(status).strip().lower():
                status = "Desatualizada"
            else:
                status = "Atual"

            results_for_ui.append({
                "id": f"r{counter:02d}",
                "score": round(score, 4),
                "fp": file_path,
                "tipo": metadata.get("tipo", "MOC"),
                "status": status,
                "votos": votos,
                "tags": tags,
                "updated": metadata.get("updated", ""),
                "conteudo": metadata.get("conteudo", ""),
            })

    return results_for_ui

# ==========================================
# HTML do Split View
# ==========================================
def build_splitview_html(results_json, query, total_docs, has_results):
    """Gera o HTML completo do split-view."""

    # Estado visual: se não há busca ainda, mostra tela inicial
    initial_screen = "true" if not has_results else "false"

    return """
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --bg: #FAFAF9; --bg-elev: #fff; --bg-sub: #F4F4F2;
  --bd: #E7E5E0; --bd-str: #D4D2CC;
  --fg: #1A1916; --fg-m: #6B6964; --fg-f: #9C9A94;
  --accent: oklch(0.55 0.13 250); --accent-soft: oklch(0.96 0.02 250); --accent-bd: oklch(0.88 0.04 250);
  --warn: oklch(0.62 0.13 65); --warn-soft: oklch(0.96 0.03 80);
  --ok: oklch(0.58 0.11 155); --ok-soft: oklch(0.96 0.03 155);
  --font: 'Inter', -apple-system, system-ui, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body { font-family: var(--font); color: var(--fg); background: var(--bg); height: 100%; overflow: hidden; }
::selection { background: var(--accent-soft); }
input[type="checkbox"] { accent-color: var(--accent); }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bd-str); border-radius: 3px; }

.pill { display: inline-flex; align-items: center; gap: 4px; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 500; white-space: nowrap; border: 1px solid transparent; margin: 0 3px 3px 0; }
.pill-accent { background: var(--accent-soft); color: oklch(0.38 0.12 250); border-color: var(--accent-bd); }
.pill-neutral { background: var(--bg-sub); color: #4A4844; border-color: var(--bd); }
.pill-warn { background: var(--warn-soft); color: oklch(0.42 0.11 65); border-color: oklch(0.85 0.06 80); }
.pill-ok { background: var(--ok-soft); color: oklch(0.38 0.10 155); border-color: oklch(0.85 0.05 155); }
.type-chip { display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px; border-radius: 4px; font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px; }
.type-moc { background: oklch(0.95 0.03 250); color: oklch(0.38 0.12 250); }
.type-juris { background: oklch(0.95 0.025 30); color: oklch(0.40 0.11 30); }
.dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; }
mark { background: rgba(59,130,246,0.18); color: inherit; padding: 0 2px; border-radius: 2px; }
.fg { font-size: 10px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: var(--fg-m); margin-bottom: 8px; }
.group-hdr { font-size: 10px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: var(--fg-m); padding: 10px 20px 6px; border-bottom: 1px solid var(--bd); background: var(--bg); display: flex; justify-content: space-between; }
.meta-label { font-size: 10px; font-weight: 600; letter-spacing: 0.8px; text-transform: uppercase; color: var(--fg-m); margin-bottom: 5px; }
.meta-label.pri { color: oklch(0.38 0.12 250); }
.content-box { background: var(--bg); border: 1px solid var(--bd); border-radius: 8px; padding: 16px 20px; font-size: 13.5px; line-height: 1.7; white-space: pre-wrap; max-width: 720px; }
.btn { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--bd); background: transparent; padding: 5px 12px; border-radius: 6px; font-size: 12px; font-family: var(--font); font-weight: 500; cursor: pointer; color: var(--fg); transition: background .1s; }
.btn:hover { background: var(--bg-sub); }

.app { display: flex; flex-direction: column; height: 100vh; }
.topbar { display: flex; align-items: center; gap: 16px; padding: 10px 24px; border-bottom: 1px solid var(--bd); background: var(--bg-elev); flex-shrink: 0; }
.body { flex: 1; display: grid; grid-template-columns: 224px 1fr 1fr; overflow: hidden; min-height: 0; }
.body.no-results { grid-template-columns: 1fr; }
.filters { border-right: 1px solid var(--bd); padding: 18px 16px; overflow-y: auto; background: var(--bg); }
.filter-group { margin-bottom: 18px; }
.filter-label { display: flex; align-items: center; gap: 8px; font-size: 13px; padding: 4px 0; cursor: pointer; }
.list { border-right: 1px solid var(--bd); overflow-y: auto; background: var(--bg-elev); }
.row { padding: 14px 18px; border-bottom: 1px solid var(--bd); cursor: pointer; border-left: 3px solid transparent; transition: background .08s; }
.row:hover { background: var(--bg-sub); }
.row.sel { background: var(--accent-soft); border-left-color: var(--accent); }
.detail { overflow-y: auto; background: var(--bg); }
.empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--fg-f); gap: 8px; text-align: center; padding: 40px; }
.toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%) translateY(60px); background: var(--fg); color: #fff; padding: 8px 20px; border-radius: 8px; font-size: 13px; font-weight: 500; opacity: 0; transition: all .25s cubic-bezier(.2,.8,.2,1); z-index: 999; pointer-events: none; }
.toast.show { opacity: 1; transform: translateX(-50%) translateY(0); }

/* Tela inicial */
.welcome { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 16px; text-align: center; }
.welcome h1 { font-size: 28px; font-weight: 600; letter-spacing: -0.5px; color: var(--fg); }
.welcome p { font-size: 14px; color: var(--fg-m); max-width: 420px; line-height: 1.6; }

/* Search */
.search-wrap { position: relative; flex: 1; max-width: 600px; }
.search-input { width: 100%; border: 1px solid var(--bd); border-radius: 8px; padding: 9px 14px 9px 38px; font-size: 14px; font-family: var(--font); color: var(--fg); background: var(--bg-elev); outline: none; transition: border-color .12s, box-shadow .12s; }
.search-input:focus { border-color: var(--accent-bd); box-shadow: 0 0 0 3px var(--accent-soft); }
.search-icon { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); color: var(--fg-m); pointer-events: none; }

/* Welcome search é maior */
.welcome .search-wrap { max-width: 520px; width: 100%; }
.welcome .search-input { padding: 14px 18px 14px 44px; font-size: 16px; border-radius: 12px; }
.welcome .search-icon { left: 16px; }
.welcome .search-hint { font-size: 11px; color: var(--fg-f); margin-top: 4px; }
</style>
</head>
<body>
<div class="app">
  <!-- Top bar -->
  <div class="topbar">
    <div style="display:flex;align-items:center;gap:10px">
      <div style="width:28px;height:28px;border-radius:6px;background:var(--fg);color:#fff;display:grid;place-items:center;font-size:14px;font-weight:700">⚖</div>
      <div>
        <div style="font-size:14px;font-weight:600">Uniformizador</div>
        <div style="font-size:10px;color:var(--fg-m);letter-spacing:0.3px;text-transform:uppercase">Gabinete · Jurisprudência</div>
      </div>
    </div>

    <form id="topSearchForm" style="position:relative;flex:1;max-width:600px;margin:0 20px;display:none">
      <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
      <input class="search-input" id="topSearchInput" placeholder="Pesquisar jurisprudência..." />
    </form>

    <div style="flex:1"></div>

    <div id="statsBar" style="font-size:11px;color:var(--fg-m);display:flex;gap:16px;align-items:center">
      <span><span style="font-family:var(--mono)">""" + str(total_docs) + """</span> docs</span>
    </div>

    <div style="display:flex;align-items:center;gap:8px;margin-left:18px">
      <div style="width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;display:grid;place-items:center;font-size:10px;font-weight:600">RC</div>
      <div style="font-size:12px"><div style="font-weight:500">Ramon C.</div><div style="font-size:10px;color:var(--fg-m)">Assessor</div></div>
    </div>
  </div>

  <!-- Body -->
  <div class="body" id="mainBody">
    <!-- Conteúdo injetado por JS -->
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const RESULTS = """ + results_json + """;
const QUERY = """ + json.dumps(query) + """;
const IS_INITIAL = """ + initial_screen + """;

// ═══════ Utils ═══════
function parseFP(fp) {
  const p = fp.split("/"); const nm = p[p.length-1].replace(/\\.md$/,"").replace(/_/g," ");
  const cat = p.length > 2 ? p[1] : "Geral"; const bc = p.slice(0,-1).join(" / ");
  return { name: nm, cat, bc };
}
function esc(s) { const d=document.createElement("div"); d.textContent=s; return d.innerHTML; }
function hl(text, q) {
  if(!text||!q) return esc(text||"");
  const terms = q.split(/\\s+/).filter(t=>t.length>2).map(t=>t.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&'));
  if(!terms.length) return esc(text);
  return esc(text).replace(new RegExp("("+terms.join("|")+")","gi"), "<mark>$1</mark>");
}
function relBar(score, w) {
  const pct = Math.round(score*100);
  const c = pct>=80?"oklch(0.58 0.11 155)":pct>=65?"oklch(0.55 0.13 250)":"oklch(0.62 0.08 260)";
  return '<span style="display:inline-flex;align-items:center;gap:8px"><span style="width:'+w+'px;height:4px;background:var(--bd);border-radius:2px;overflow:hidden"><span style="display:block;height:100%;width:'+pct+'%;background:'+c+';border-radius:2px"></span></span><span style="font-family:var(--mono);font-size:11px;color:var(--fg-m)">'+pct+'%</span></span>';
}
function typeChip(tipo) {
  const j = tipo.toLowerCase().includes("juris");
  const cls = j ? "type-juris" : "type-moc";
  const dot = j ? "oklch(0.60 0.13 30)" : "var(--accent)";
  return '<span class="type-chip '+cls+'"><span class="dot" style="background:'+dot+'"></span>'+(j?"Jurisprudência":"MOC")+'</span>';
}
function pills(arr, tone) { return arr.map(v=>'<span class="pill pill-'+tone+'">'+esc(v)+'</span>').join(""); }
function showToast(msg) {
  const el=document.getElementById("toast"); el.textContent=msg; el.classList.add("show");
  setTimeout(()=>el.classList.remove("show"),1800);
}

// ═══════ Streamlit comm ═══════
function triggerSearch(query) {
  // Envia a query para o Streamlit via query params
  const url = new URL(window.parent.location.href);
  url.searchParams.set("q", query);
  window.parent.history.replaceState({}, "", url);
  // Força rerun do Streamlit
  window.parent.postMessage({ type: "streamlit:setComponentValue", value: query }, "*");
  // Fallback: recarrega
  window.parent.location.href = url.toString();
}

// ═══════ State ═══════
let selectedId = RESULTS.length ? RESULTS[0].id : null;
let filters = { tipos: new Set(["MOC","Jurisprudência"]), status: new Set(["Atual"]), threshold: 50, tags: new Set() };
const allTags = Array.from(new Set(RESULTS.flatMap(r=>r.tags||[])));

function getFiltered() {
  return RESULTS.filter(r =>
    filters.tipos.has(r.tipo) &&
    filters.status.has(r.status) &&
    Math.round(r.score*100) >= filters.threshold &&
    (filters.tags.size===0 || (r.tags||[]).some(t=>filters.tags.has(t)))
  );
}

// ═══════ Render ═══════
function renderWelcome() {
  const body = document.getElementById("mainBody");
  body.className = "body no-results";
  body.innerHTML = `
    <div class="welcome">
      <div style="width:56px;height:56px;border-radius:14px;background:var(--fg);color:#fff;display:grid;place-items:center;font-size:28px;font-weight:700">⚖</div>
      <h1>Pesquisa Jurisprudencial</h1>
      <p>Busque por temas, teses ou palavras-chave. A pesquisa semântica encontra resultados por significado, não apenas por palavras exatas.</p>
      <form onsubmit="event.preventDefault();triggerSearch(this.querySelector('input').value)" class="search-wrap" style="max-width:520px;width:100%;position:relative">
        <svg class="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="left:16px"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>
        <input class="search-input" style="padding:14px 18px 14px 44px;font-size:16px;border-radius:12px" placeholder="Ex: taxa de lixo progressiva" autofocus />
      </form>
      <div style="font-size:11px;color:var(--fg-f);margin-top:4px">Pressione Enter para pesquisar</div>
    </div>`;
}

function renderSplitView() {
  const body = document.getElementById("mainBody");
  body.className = "body";
  body.style.gridTemplateColumns = "224px 1fr 1fr";
  body.innerHTML = '<div class="filters" id="filtersPanel"></div><div class="list" id="listPanel"></div><div class="detail" id="detailPanel"></div>';

  // Mostra busca no topbar
  document.getElementById("topSearchForm").style.display = "block";
  document.getElementById("topSearchInput").value = QUERY;
  document.getElementById("topSearchForm").onsubmit = function(e) {
    e.preventDefault();
    triggerSearch(document.getElementById("topSearchInput").value);
  };

  renderFilters();
  renderList();
}

function renderFilters() {
  const el = document.getElementById("filtersPanel");
  const typeCounts = { MOC: RESULTS.filter(r=>r.tipo==="MOC").length, "Jurisprudência": RESULTS.filter(r=>r.tipo==="Jurisprudência").length };
  el.innerHTML = `
    <div class="filter-group"><div class="fg">Tipo</div>
      ${["MOC","Jurisprudência"].map(t=>`<label class="filter-label"><input type="checkbox" ${filters.tipos.has(t)?"checked":""} onchange="toggleFilter('tipos','${t}')">${t}<span style="margin-left:auto;font-family:var(--mono);font-size:11px;color:var(--fg-f)">${typeCounts[t]||0}</span></label>`).join("")}
    </div>
    <div class="filter-group"><div class="fg">Status</div>
      <label class="filter-label"><input type="checkbox" ${filters.status.has("Atual")?"checked":""} onchange="toggleFilter('status','Atual')">Atual</label>
      <label class="filter-label"><input type="checkbox" ${filters.status.has("Desatualizada")?"checked":""} onchange="toggleFilter('status','Desatualizada')" style="accent-color:var(--warn)">Desatualizada</label>
    </div>
    <div class="filter-group"><div class="fg">Limiar de relevância</div>
      <input type="range" min="30" max="95" value="${filters.threshold}" oninput="setThreshold(+this.value)" style="width:100%;accent-color:var(--accent)">
      <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--fg-m);margin-top:4px"><span>30%</span><span style="font-family:var(--mono);font-weight:600;color:var(--fg)">≥ ${filters.threshold}%</span><span>95%</span></div>
    </div>
    <div class="filter-group"><div class="fg">Tags</div>
      <div style="display:flex;flex-wrap:wrap;gap:5px">
        ${allTags.map(t=>`<button onclick="toggleTag('${t.replace(/'/g,"\\'")}')" style="display:inline-flex;font-size:10px;padding:3px 9px;border-radius:4px;border:1px solid ${filters.tags.has(t)?'var(--accent-bd)':'var(--bd)'};background:${filters.tags.has(t)?'var(--accent-soft)':'transparent'};color:${filters.tags.has(t)?'oklch(0.38 0.12 250)':'var(--fg)'};cursor:pointer;font-family:var(--font);font-weight:500">${esc(t)}</button>`).join("")}
      </div>
    </div>`;
}

function renderList() {
  const el = document.getElementById("listPanel");
  const items = getFiltered();
  if (!items.find(r=>r.id===selectedId) && items.length) selectedId = items[0].id;

  const grouped = {};
  items.forEach(r=>{ (grouped[r.tipo]=grouped[r.tipo]||[]).push(r); });

  let html = `<div style="padding:14px 18px;border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between">
    <div><div style="font-size:13px;font-weight:600">${items.length} resultado${items.length!==1?"s":""}</div>
    <div style="font-size:11px;color:var(--fg-m);margin-top:2px">para "<i style="color:var(--fg)">${esc(QUERY)}</i>"</div></div>
    <div style="font-size:10px;color:var(--fg-f);font-family:var(--mono)">↑↓ navegar</div></div>`;

  if (!items.length) {
    html += '<div style="padding:40px;text-align:center;color:var(--fg-f)"><div style="font-size:14px;font-weight:500;color:var(--fg-m)">Nenhum resultado</div><div style="font-size:12px">Ajuste os filtros ou o limiar</div></div>';
  } else {
    for (const [tipo, rs] of Object.entries(grouped)) {
      html += `<div class="group-hdr"><span>${tipo}</span><span style="font-family:var(--mono)">${rs.length}</span></div>`;
      for (const r of rs) {
        const {name,bc} = parseFP(r.fp);
        const isD = r.status==="Desatualizada";
        html += `<div class="row${r.id===selectedId?' sel':''}" onclick="selectRow('${r.id}')">
          <div style="margin-bottom:4px;display:flex;align-items:center;gap:8px">${relBar(r.score,44)}${isD?'<span class="pill pill-warn">⚠ Desatualizada</span>':''}</div>
          <div style="font-size:13px;font-weight:600;letter-spacing:-0.1px;margin-bottom:2px;line-height:1.3">${hl(name,QUERY)}</div>
          <div style="font-family:var(--mono);font-size:11px;color:var(--fg-f);margin-bottom:6px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${esc(bc)}</div>
          <div style="font-size:12px;color:var(--fg-m);line-height:1.5;margin-bottom:6px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">${hl((r.conteudo||"").split("\\n")[0].slice(0,140)+"…",QUERY)}</div>
          <div>${pills((r.votos||[]).slice(0,2),"accent")}${(r.votos||[]).length>2?'<span class="pill pill-neutral">+'+(r.votos.length-2)+'</span>':''}</div>
        </div>`;
      }
    }
  }
  el.innerHTML = html;
  renderDetail();
}

function renderDetail() {
  const el = document.getElementById("detailPanel");
  const r = RESULTS.find(x=>x.id===selectedId);
  if (!r) { el.innerHTML = '<div class="empty"><div style="font-size:14px;font-weight:500;color:var(--fg-m)">Selecione um resultado</div></div>'; return; }
  const {name,bc} = parseFP(r.fp);
  const isD = r.status==="Desatualizada";
  const cit = name+" — "+r.tipo+". Votos: "+(r.votos||[]).join("; ")+". Fonte: Gabinete ("+r.fp+").";
  el.innerHTML = `
    <div style="padding:18px 28px 16px;border-bottom:1px solid var(--bd);background:var(--bg-elev)">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
        ${typeChip(r.tipo)}
        ${isD?'<span class="pill pill-warn">⚠ Desatualizada</span>':'<span class="pill pill-ok">✓ Atual</span>'}
        <span style="flex:1"></span>
        ${relBar(r.score,72)}
      </div>
      <div style="font-family:var(--mono);font-size:11px;color:var(--fg-f);margin-bottom:6px">${esc(bc)}</div>
      <h2 style="margin:0;font-size:22px;font-weight:600;letter-spacing:-0.4px;line-height:1.25">${hl(name,QUERY)}</h2>
      <div style="display:flex;gap:8px;margin-top:14px;flex-wrap:wrap">
        <button class="btn" onclick="navigator.clipboard.writeText(\`${cit.replace(/`/g,'\\`')}\`);showToast('Citação copiada')">📋 Copiar citação</button>
        <button class="btn">⬇ Exportar</button>
        <button class="btn">☆ Favoritar</button>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;padding:14px 28px;background:var(--bg-elev);border-bottom:1px solid var(--bd)">
      <div><div class="meta-label pri">Votos aplicados</div>${(r.votos||[]).length?pills(r.votos,"accent"):'<span style="font-size:12px;color:var(--fg-f);font-style:italic">Nenhum</span>'}</div>
      <div><div class="meta-label">Atualizado</div><span style="font-family:var(--mono);font-size:13px">${r.updated||"—"}</span></div>
      <div style="grid-column:1/-1;padding-top:12px"><div class="meta-label">Tags</div>${pills((r.tags||[]).map(t=>"#"+t),"neutral")}</div>
    </div>
    <div style="padding:24px 28px 40px">
      <div class="meta-label" style="margin-bottom:12px">Conteúdo da nota</div>
      <div class="content-box">${hl(r.conteudo||"",QUERY)}</div>
    </div>`;
}

function selectRow(id) { selectedId=id; renderList(); }
function toggleFilter(key,val) { filters[key].has(val)?filters[key].delete(val):filters[key].add(val); renderFilters(); renderList(); }
function setThreshold(v) { filters.threshold=v; renderFilters(); renderList(); }
function toggleTag(t) { filters.tags.has(t)?filters.tags.delete(t):filters.tags.add(t); renderFilters(); renderList(); }

// Keyboard nav
document.addEventListener("keydown", e => {
  if (IS_INITIAL) return;
  if (document.activeElement && document.activeElement.tagName === "INPUT") return;
  const items = getFiltered();
  if (e.key==="ArrowDown"||e.key==="ArrowUp"||e.key==="j"||e.key==="k") {
    e.preventDefault();
    const idx = items.findIndex(r=>r.id===selectedId);
    const next = (e.key==="ArrowDown"||e.key==="j") ? Math.min(idx+1,items.length-1) : Math.max(idx-1,0);
    if (items[next]) { selectedId=items[next].id; renderList(); }
  }
  if (e.key==="/") { e.preventDefault(); document.getElementById("topSearchInput")?.focus(); }
});

// ═══════ Init ═══════
if (IS_INITIAL) {
  renderWelcome();
} else {
  renderSplitView();
}
</script>
</body>
</html>
"""


# ==========================================
# App Principal
# ==========================================
def main():
    if not check_password():
        return

    # Conexão Pinecone
    try:
        pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
        index_name = st.secrets.get("PINECONE_INDEX_NAME", "default-index")
        index = pc.Index(index_name)
    except Exception as e:
        st.error("Falha ao conectar ao Pinecone.")
        st.exception(e)
        return

    # Total de docs
    try:
        stats = index.describe_index_stats()
        total_docs = stats.get("total_vector_count", 0)
    except:
        total_docs = 0

    # Lê query dos query params (vem do HTML via redirect)
    query_text = st.query_params.get("q", "")

    if query_text:
        try:
            results = do_search(query_text, index)
            results_json = json.dumps(results, ensure_ascii=False)
            html = build_splitview_html(results_json, query_text, total_docs, has_results=True)
        except Exception as e:
            st.error(f"Erro na pesquisa: {e}")
            return
    else:
        html = build_splitview_html("[]", "", total_docs, has_results=False)

    st.components.v1.html(html, height=800, scrolling=False)


if __name__ == "__main__":
    main()
