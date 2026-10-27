import sys, os, json, re, html, shutil

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\aksha\DSAI_Projects"
TRANSCRIPT_LOG = r"C:\Users\aksha\.gemini\antigravity\brain\e69e39b2-2ddd-4d0a-ad5c-ac754f22de89\.system_generated\logs\transcript_full.jsonl"
MD_PATH = os.path.join(BASE_DIR, "transcript.md")
HTML_PATH = os.path.join(BASE_DIR, "transcript.html")

# 1. Titles and Summaries for the 16 new turns (Turns 47 to 62)
NEW_TURNS_META = [
    {
        "num": 47,
        "title": "Formulary Size vs. Training Samples: Clarifying 50 Drug Classes vs. 8,000 Dataset Prescriptions",
        "summary": "Clarifying that the model was trained on 8,000 synthetic clinical prescriptions across 50 OPD drug classes rather than 50 data points, detailing split distributions and statistical power."
    },
    {
        "num": 48,
        "title": "Architectural Scaling Roadmap: Expanding from 50 OPD Drugs to Full NLEM 2022 (384 Medicines)",
        "summary": "Analyzing the architectural and computational impact of scaling the formulary from 50 to 384 drugs, including vocabulary expansion, dimensional capacity, embedding overhead, and synthetic combinatorial coverage."
    },
    {
        "num": 49,
        "title": "Training Dynamics & Epoch Budgets: Loss Convergence, Two-Stage Guarantees & Overfitting Bounds",
        "summary": "Explaining why 8 epochs was originally chosen based on loss saturation (<0.005), learning rate annealing schedules, and the separation of probabilistic translation from deterministic DFA decoding."
    },
    {
        "num": 50,
        "title": "Training Ceiling & Early Stopping Mechanics: 15-Epoch Bound with Patience (k=3) and Delta Criterion",
        "summary": "Proposing an upper ceiling of 15 epochs guarded by validation loss monitoring, min-delta threshold (1e-4), and patience-based early stopping to prevent overtraining while maximizing accuracy."
    },
    {
        "num": 51,
        "title": "Implementation of Early Stopping & Project Documentation Synchronization (train.py, WALKTHROUGH.md, PLAN.md)",
        "summary": "Updating train.py with configurable early stopping, 15-epoch ceiling, best-checkpoint preservation, and synchronizing project walkthrough and sprint plan specifications."
    },
    {
        "num": 52,
        "title": "Checkpoint Isolation Strategy: Preserving 50-Drug Baseline vs. Full Formulary Architecture",
        "summary": "Formulating a dual-checkpoint architecture strategy to prevent overwriting the pristine 91.33% 50-drug model while establishing separate paths for full NLEM training."
    },
    {
        "num": 53,
        "title": "15-Epoch Training Execution on 50-Drug Baseline: Early Stopping Verification at Epoch 10 (91.33% EM)",
        "summary": "Executing the 15-epoch training in the umc301 environment, observing automatic early stopping trigger at epoch 10 after validation loss plateaued, and confirming 91.33% Exact Match."
    },
    {
        "num": 54,
        "title": "Formulary Expansion Strategy: Retaining Pristine 91.33% Checkpoints and Scaling Path Confirmation",
        "summary": "Confirming preservation of clinical_transformer_best_50drugs.pt (91.33% EM) and establishing the 8-epoch training protocol for the full 384-drug NLEM formulary."
    },
    {
        "num": 55,
        "title": "Checkpoint Redundancy Audit: Explaining Triplicate Artifacts (best, best_50drugs, epoch4)",
        "summary": "Clarifying checkpoint saving dynamics during training runs versus explicit post-run named backups and disk-space footprint."
    },
    {
        "num": 56,
        "title": "Autonomous NLEM 2022 Construction: 384-Drug Formulary, 40,000 Prescription Dataset & 1,696-Token Vocab",
        "summary": "Executing build_nlem_formulary_full.py and dataset_generator_full.py to generate the 384-drug master JSON, 40,000 synthetic clinical examples, and expanded tokenizer vocabulary."
    },
    {
        "num": 57,
        "title": "Standalone Full NLEM Training Engine: Implementation of src/train_full_nlem.py",
        "summary": "Creating a dedicated, isolated training engine with dynamic vocabulary resolution, separate checkpoint destination, and full metric logging."
    },
    {
        "num": 58,
        "title": "Full NLEM 2022 Training Run: 8 Epochs on 40,000 Samples with Isolated Model Checkpoint",
        "summary": "Executing full NLEM training across 40k samples (21.41 minutes), achieving 0.0009 validation loss and 57.50% raw Exact Match, preserved as clinical_transformer_full_nlem.pt."
    },
    {
        "num": 59,
        "title": "Error Analysis: Deconstructing the 57.5% Raw EM Drop & Tokenizer % Bug Root Cause",
        "summary": "Investigating why raw EM dropped despite lower loss, discovering the regex % tokenization bug on liquid dosages (e.g. 5% Dextrose) that accounted for ~6.5% error, and demonstrating DFA masking recovery."
    },
    {
        "num": 60,
        "title": "Architectural Defense of the Two-Stage Paradigm: Combinatorial Scaling & Intent Extraction",
        "summary": "Pedagogical defense of why 57.5% raw EM validates the two-stage compiler architecture: Transformer extracts messy clinical intent while DFA guarantees 0% drug hallucination and 100% FHIR validity."
    },
    {
        "num": 61,
        "title": "Slot-Level Translation Accuracy Audit: Drug Name (78.2%), Frequency (90.0%), Duration (84.7%) & Condition (89.6%)",
        "summary": "Quantifying granular clinical slot accuracies across the 384-drug dataset and demonstrating that end-to-end FHIR compliance exceeds 88% with constrained decoding."
    },
    {
        "num": 62,
        "title": "Dual Application Architecture: Standalone Streamlit Dashboards for 50-Drug and 384-Drug Deployments",
        "summary": "Implementing app_50drugs.py and app_full_nlem.py with custom formulary parameters in DFA and FHIR modules, verifying both applications and all 19 unit tests passing."
    }
]

# 2. Titles for former turns 47-60 (now turns 63-76)
RENUMBERED_TURNS_META = [
    {
        "old_num": 47, "num": 63,
        "title": "Viva Defense Boundary: Disentangling Project 2 (Edge Compiler) vs. Project 3 (Indic Tokenizer & LoRA SFT)",
        "summary": "Clarifying that MinHash, BPE, BPB, Chinchilla, LoRA, and NF4 belong to Project 3, while Project 2 centers on DFA logit masks and HL7 FHIR lowering."
    },
    {
        "old_num": 48, "num": 64,
        "title": "Full UI/UX Modernization Plan: Designing the Clinical Compiler Web Workstation",
        "summary": "Detailed design specification for transitioning from basic widgets to a dual-pane responsive medical workstation with dark/light themes."
    },
    {
        "old_num": 49, "num": 65,
        "title": "Runtime Environment Isolation: Validating the umc301 Conda Stack",
        "summary": "Diagnosing PyTorch system-level vs conda environment conflicts and enforcing explicit python interpreter paths."
    },
    {
        "old_num": 50, "num": 66,
        "title": "Streamlit Theming & Academic De-Branding: Neutral Styling & Course Content Removal",
        "summary": "Sanitizing application branding to ensure zero academic trace and deploying modern slate CSS tokens."
    },
    {
        "old_num": 51, "num": 67,
        "title": "Professionalization Overhaul: Emoji Removal, Typography Standards & Design System",
        "summary": "Strict typography hygiene replacing decorative emojis with clinical SVG iconography and tabular density."
    },
    {
        "old_num": 52, "num": 68,
        "title": "Architectural Pivot from Streamlit to Custom Web App (Starlette + Pure HTML/CSS/JS)",
        "summary": "Decoupling monolithic Streamlit execution into a 1.2ms async Starlette backend and pure vanilla JS client."
    },
    {
        "old_num": 53, "num": 69,
        "title": "Defending Clinical Entropy & DFA Out-Of-Vocabulary (OOV) Resilience in Viva",
        "summary": "Viva defense narrative proving why deterministic DFA state machine masking eliminates generative hallucinations."
    },
    {
        "old_num": 54, "num": 70,
        "title": "Clarifying Terminal Sandbox Permissions vs. File Modifications",
        "summary": "Explaining terminal sandbox boundary constraints vs client edge deployment security."
    },
    {
        "old_num": 55, "num": 71,
        "title": "Root Cause Diagnosis of Antigravity Rule Parsing Failure",
        "summary": "Auditing why rule parsing failed across concurrent agents and implementing file-based contract protocols."
    },
    {
        "old_num": 56, "num": 72,
        "title": "Multi-Agent Architecture: Static YAML Parsers vs. File-Based Tool Routing",
        "summary": "Architectural design comparing static agent manifest parsing against persistent filesystem signal hooks."
    },
    {
        "old_num": 57, "num": 73,
        "title": "Implementation of Dual-Agent Configuration (trigger: always_on & CLAUDE.md)",
        "summary": "Implementing dual-agent workspace automation configs guaranteeing continuous rule synchronization."
    },
    {
        "old_num": 58, "num": 74,
        "title": "Comprehensive Clinical Test Suite: Formulary Resolution, Schedules & Edge Cases",
        "summary": "Testing complex real-world multi-token drug schedules, OOV words, abbreviations, and pediatric dosages."
    },
    {
        "old_num": 59, "num": 75,
        "title": "Root Cause Analysis of 2-Item Generation Cap: Prior Bias & Context Arithmetic",
        "summary": "Mathematical root-cause breakdown of sequence truncation on multi-item clinical prescriptions."
    },
    {
        "old_num": 60, "num": 76,
        "title": "Implementation of Approach A: Arbitrary Multi-Line Statement Decomposition in server.py",
        "summary": "Implementing Approach A statement decomposition in the Starlette API to cleanly compile unbounded multi-line orders."
    }
]

def extract_turns():
    with open(TRANSCRIPT_LOG, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    user_indices = [i for i, step in enumerate(lines) if step.get('type') == 'USER_INPUT']
    turns = []
    
    for i in range(16):
        u_idx = user_indices[i]
        next_u_idx = user_indices[i+1]
        u_raw = lines[u_idx].get('content', '')
        m = re.search(r'<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>', u_raw, re.DOTALL)
        u_text = m.group(1).strip() if m else u_raw.strip()

        # Handle Turn 56 (index 9) special quote formatting
        if i == 9 and "@[Quote]" in u_text:
            u_text = '> "Would you like me to delete that redundant backup file now?"\n\nNo for now keep it, and\n\n> "Step 1: creating the full 384-drug formulary and dataset generator?"\n\nlets proceed with this.'

        responses = []
        for step in lines[u_idx+1:next_u_idx]:
            if step.get('type') == 'PLANNER_RESPONSE':
                c = step.get('content', '').strip()
                if c and not c.startswith('Looking for ') and not c.startswith('Waiting for '):
                    responses.append(c)

        resp_text = "\n\n".join(responses)
        turns.append({
            "num": 47 + i,
            "title": NEW_TURNS_META[i]["title"],
            "summary": NEW_TURNS_META[i]["summary"],
            "user_text": u_text,
            "asst_text": resp_text
        })
    return turns

def markdown_to_html(md_text):
    lines = md_text.split('\n')
    output = []
    in_code_block = False
    code_lang = ''
    code_lines = []
    in_table = False
    table_lines = []
    in_ul = False
    in_ol = False
    in_blockquote = False
    blockquote_lines = []
    paragraph_lines = []

    def flush_paragraph():
        nonlocal paragraph_lines
        if paragraph_lines:
            text = ' '.join(paragraph_lines).strip()
            if text:
                output.append(f'<p>{format_inlines(text)}</p>')
            paragraph_lines = []

    def flush_list():
        nonlocal in_ul, in_ol
        if in_ul:
            output.append('</ul>')
            in_ul = False
        if in_ol:
            output.append('</ol>')
            in_ol = False

    def flush_blockquote():
        nonlocal in_blockquote, blockquote_lines
        if in_blockquote:
            bq_text = ' '.join(blockquote_lines)
            output.append(f'<blockquote><p>{format_inlines(bq_text)}</p></blockquote>')
            in_blockquote = False
            blockquote_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if in_table and table_lines:
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.strip().strip('|').split('|')]
                rows.append(cells)
            
            if rows:
                table_html = ['<div class="table-wrapper"><table>']
                header_cells = rows[0]
                table_html.append('<thead><tr>')
                for c in header_cells:
                    table_html.append(f'<th>{format_inlines(c)}</th>')
                table_html.append('</tr></thead>')
                start_body = 1
                if len(rows) > 1 and all(set(c).issubset({'-', ':', ' '}) for c in rows[1]):
                    start_body = 2
                
                table_html.append('<tbody>')
                for r in rows[start_body:]:
                    table_html.append('<tr>')
                    for c in r:
                        table_html.append(f'<td>{format_inlines(c)}</td>')
                    table_html.append('</tr>')
                table_html.append('</tbody></table></div>')
                output.append(''.join(table_html))
            in_table = False
            table_lines = []

    def format_inlines(text):
        code_tokens = []
        def save_code(m):
            code_tokens.append(m.group(1))
            return f'___CODE_{len(code_tokens)-1}___'
        text = re.sub(r'`([^`]+)`', save_code, text)

        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        for idx, ct in enumerate(code_tokens):
            ct_escaped = html.escape(ct)
            text = text.replace(f'___CODE_{idx}___', f'<code>{ct_escaped}</code>')

        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<strong><em>\1</em></strong>', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" target="_blank">\1</a>', text)

        return text

    for line in lines:
        stripped = line.strip()

        if stripped.startswith('```'):
            if in_code_block:
                code_content = html.escape('\n'.join(code_lines))
                lang = code_lang if code_lang else 'text'
                output.append(f'<div class="code-block-wrapper"><div class="code-header"><span class="code-lang">{lang}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy</button></div><pre><code class="language-{lang}">{code_content}</code></pre></div>')
                in_code_block = False
                code_lines = []
                code_lang = ''
            else:
                flush_paragraph()
                flush_list()
                flush_blockquote()
                flush_table()
                in_code_block = True
                code_lang = stripped[3:].strip()
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        if stripped.startswith('|') and stripped.endswith('|'):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            flush_table()

        if stripped.startswith('>'):
            flush_paragraph()
            flush_list()
            in_blockquote = True
            blockquote_lines.append(stripped.lstrip('>').strip())
            continue
        elif in_blockquote and stripped != '':
            blockquote_lines.append(stripped)
            continue
        elif in_blockquote and stripped == '':
            flush_blockquote()

        if stripped == '':
            flush_paragraph()
            flush_list()
            flush_blockquote()
            flush_table()
            continue

        if re.match(r'^(?:---|\*\*\*|___)$', stripped):
            flush_paragraph()
            flush_list()
            flush_blockquote()
            flush_table()
            output.append('<hr class="divider"/>')
            continue

        h_match = re.match(r'^(#{1,6})\s+(.*)$', stripped)
        if h_match:
            flush_paragraph()
            flush_list()
            flush_blockquote()
            flush_table()
            level = len(h_match.group(1))
            h_text = format_inlines(h_match.group(2))
            output.append(f'<h{level}>{h_text}</h{level}>')
            continue

        ul_match = re.match(r'^[\*\-\+]\s+(.*)$', stripped)
        if ul_match:
            flush_paragraph()
            if in_ol:
                flush_list()
            if not in_ul:
                output.append('<ul>')
                in_ul = True
            item_text = format_inlines(ul_match.group(1))
            output.append(f'<li>{item_text}</li>')
            continue

        ol_match = re.match(r'^\d+\.\s+(.*)$', stripped)
        if ol_match:
            flush_paragraph()
            if in_ul:
                flush_list()
            if not in_ol:
                output.append('<ol>')
                in_ol = True
            item_text = format_inlines(ol_match.group(1))
            output.append(f'<li>{item_text}</li>')
            continue

        flush_list()
        flush_blockquote()
        flush_table()
        paragraph_lines.append(stripped)

    if in_code_block:
        code_content = html.escape('\n'.join(code_lines))
        lang = code_lang if code_lang else 'text'
        output.append(f'<div class="code-block-wrapper"><div class="code-header"><span class="code-lang">{lang}</span><button class="copy-code-btn" onclick="copyCode(this)">Copy</button></div><pre><code class="language-{lang}">{code_content}</code></pre></div>')
    flush_paragraph()
    flush_list()
    flush_blockquote()
    flush_table()

    return '\n'.join(output)

def generate_updated_markdown(new_turns):
    with open(MD_PATH, 'r', encoding='utf-8') as f:
        md_text = f.read()

    matches = list(re.finditer(r'^## Turn 47:', md_text, re.MULTILINE))
    if not matches:
        raise ValueError("Could not find ## Turn 47: in transcript.md")
    body_turn47_pos = matches[-1].start()

    front_part = md_text[:body_turn47_pos]
    old_body_part = md_text[body_turn47_pos:]

    # 1. Update TOC in front_part
    toc_part4_match = re.search(r'### Part IV:.*?(?=\n\n(?:---|## Turn 1:|\Z))', front_part, re.DOTALL)
    if not toc_part4_match:
        toc_part4_match = re.search(r'### Part IV:.*?\* \[Turn 60:[^\n]+\]\(#turn-60\)', front_part, re.DOTALL)
    
    if not toc_part4_match:
        raise ValueError("Could not locate Part IV TOC in transcript.md")

    part4_toc_lines = [
        "### Part IV: Project 2 Formulary Scaling (50 to 384 NLEM Drugs), Training Dynamics & Dual Dashboards (Turns 47–62)"
    ]
    for t in new_turns:
        part4_toc_lines.append(f"* [Turn {t['num']}: {t['title']}](#turn-{t['num']})")

    part5_toc_lines = [
        "### Part V: Project 2 Enterprise UI/UX, Multi-Line Compiler & Cross-Agent Tools (Turns 63–76)"
    ]
    for rm in RENUMBERED_TURNS_META:
        part5_toc_lines.append(f"* [Turn {rm['num']}: {rm['title']}](#turn-{rm['num']})")

    new_toc = "\n".join(part4_toc_lines) + "\n\n" + "\n".join(part5_toc_lines)
    updated_front_part = front_part[:toc_part4_match.start()] + new_toc + front_part[toc_part4_match.end():]

    # 2. Build Part IV Body
    part4_body_lines = [
        "# Part IV: Project 2 Formulary Scaling (50 to 384 NLEM Drugs), Training Dynamics & Dual Dashboards (Turns 47–62)",
        "",
        "---",
        ""
    ]
    for t in new_turns:
        part4_body_lines.append(f"## Turn {t['num']}: {t['title']} {{#turn-{t['num']}}}")
        part4_body_lines.append("**Phase:** Part IV: P2 Formulary Scaling | **Source:** `Antigravity Workspace Session`  \n")
        part4_body_lines.append("### 👤 User Prompt\n")
        part4_body_lines.append(t['user_text'] + "\n")
        part4_body_lines.append("### 🤖 Antigravity Assistant Response\n")
        part4_body_lines.append(t['asst_text'] + "\n")
        part4_body_lines.append("---\n")

    part4_body = "\n".join(part4_body_lines)

    # 3. Renumber old Part IV Body to become Part V Body (Turns 63 to 76)
    part5_body = old_body_part
    for old_n in range(60, 46, -1):
        new_n = old_n + 16
        part5_body = re.sub(rf'^## Turn {old_n}: (.*?)\{{#turn-{old_n}\}}$', rf'## Turn {new_n}: \1{{#turn-{new_n}}}', part5_body, flags=re.MULTILINE)
        part5_body = re.sub(rf'Phase:\*\* Part IV:', rf'Phase:** Part V:', part5_body)

    part5_body_header = "# Part V: Project 2 Enterprise UI/UX, Multi-Line Compiler & Cross-Agent Tools (Turns 63–76)\n\n---\n\n"
    part5_body = part5_body_header + part5_body

    full_updated_md = updated_front_part.rstrip() + "\n\n---\n\n" + part4_body.strip() + "\n\n---\n\n" + part5_body.strip() + "\n"
    return full_updated_md

def generate_updated_html(new_turns, all_titles):
    with open(HTML_PATH, 'r', encoding='utf-8') as f:
        html_text = f.read()

    pos_sidebar = html_text.find('<aside class="sidebar"')
    pre_sidebar = html_text[:pos_sidebar]

    # Build Sidebar Navigation
    sidebar_lines = [
        '<aside class="sidebar" id="sidebar">',
        '    <div class="sidebar-header">',
        '        <div class="sidebar-title">',
        '            <span>📑</span> Transcript Index',
        '        </div>',
        '        <div class="sidebar-subtitle">IISc CDS UMC301 | 76 Turns</div>',
        '    </div>',
        '    <div class="sidebar-search-box">',
        '        <input type="text" class="sidebar-search-input" id="sidebarSearch" placeholder="Filter turns..." oninput="filterSidebar(this.value)">',
        '    </div>',
        '    <nav class="sidebar-scroll" id="sidebarList">'
    ]

    sections = [
        ("PART I: CLI FOUNDATIONS", 1, 10, "CLI"),
        ("PART II: WEB / IDE PORTFOLIO", 11, 28, "IDE"),
        ("PART III: PROJECT 2 CORE ARCHITECTURE", 29, 46, "P2"),
        ("PART IV: FORMULARY SCALING &amp; TRAINING", 47, 62, "SCALE"),
        ("PART V: ENTERPRISE UI &amp; MULTI-LINE COMPILER", 63, 76, "UI")
    ]

    for sec_title, start_n, end_n, badge_tag in sections:
        sidebar_lines.append(f'        <div class="sidebar-section-title">{sec_title}</div>')
        for n in range(start_n, end_n + 1):
            t_title = all_titles[n - 1]
            sidebar_lines.append(f'        <a href="#turn-{n}" class="sidebar-item" id="nav-item-{n}" data-turn="{n}">')
            sidebar_lines.append(f'            <span class="nav-turn-num">{n}</span>')
            sidebar_lines.append(f'            <div class="nav-info">')
            sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
            sidebar_lines.append(f'                <span class="nav-turn-tag">{badge_tag}</span>')
            sidebar_lines.append(f'            </div>')
            sidebar_lines.append(f'        </a>')

    sidebar_lines.append('    </nav>')
    sidebar_lines.append('</aside>')
    sidebar_html = "\n".join(sidebar_lines)

    # Main wrapper top, hero banner, filter bar
    middle_html = """
    <!-- Main Container -->
    <div class="main-wrapper">
        <!-- Sticky Navbar -->
        <header class="top-navbar">
            <div class="top-nav-left">
                <button class="btn mobile-menu-btn" onclick="toggleSidebar()">☰ Menu</button>
                <h1 class="doc-title">Master Discussion Transcript</h1>
                <span class="doc-tag">UMC301 Applied DS & AI</span>
            </div>
            <div class="top-nav-right">
                <button class="btn" onclick="expandAllResponses()">▼ Expand All</button>
                <button class="btn" onclick="collapseAllResponses()">▲ Collapse All</button>
                <button class="btn" id="themeBtn" onclick="toggleTheme()">☀️ Light</button>
            </div>
        </header>

        <!-- Main Content -->
        <main class="content-container">
            <!-- Hero Summary Card -->
            <section class="hero-banner">
                <h2 class="hero-title">IISc UMC301 Master Project & Viva Preparation Transcript</h2>
                <p class="hero-desc">
                    Comprehensive chronological record of strategic architecture, algorithmic design decisions, statistical failure-mode autopsies, and mock viva defense across 4 foundational AI pillars (Clean Energy Tabular DS, Clinical Shorthand-to-FHIR Compiler, Hindi & Telugu SLM Token Fertility, and GitBot Spec-Driven Agent).
                </p>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-num">76</div>
                        <div class="stat-label">Total Conversation Turns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">10</div>
                        <div class="stat-label">Part I: CLI Session Turns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">18</div>
                        <div class="stat-label">Part II: Web/IDE Discussion</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">18</div>
                        <div class="stat-label">Part III: Project 2 Core</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">16</div>
                        <div class="stat-label">Part IV: P2 Formulary Scaling</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">14</div>
                        <div class="stat-label">Part V: Enterprise UI & Compiler</div>
                    </div>
                </div>
            </section>

            <!-- Filter Bar -->
            <div class="filter-bar">
                <button class="filter-pill active" onclick="filterCards('all', this)">All Turns (76)</button>
                <button class="filter-pill" onclick="filterCards('cli', this)">Part I: CLI Session (10)</button>
                <button class="filter-pill" onclick="filterCards('web', this)">Part II: Web/IDE (18)</button>
                <button class="filter-pill" onclick="filterCards('p2', this)">Part III: Project 2 Core (18)</button>
                <button class="filter-pill" onclick="filterCards('p4', this)">Part IV: P2 Formulary Scaling (16)</button>
                <button class="filter-pill" onclick="filterCards('p5', this)">Part V: Enterprise UI & Multi-Line (14)</button>
            </div>

            <!-- Turn Cards List -->
            <div id="turnsList">
    """

    # Turn Cards 1 to 46 from existing HTML
    pos_c1 = html_text.find('<article class="turn-card" id="turn-1"')
    pos_c46 = html_text.find('<article class="turn-card" id="turn-46"')
    pos_c46_end = html_text.find('</article>', pos_c46) + len('</article>')
    cards_1_to_46 = html_text[pos_c1:pos_c46_end]

    # Update cards 11 to 28 data-part to "web" so Part II filter works
    for n in range(11, 29):
        cards_1_to_46 = cards_1_to_46.replace(
            f'<article class="turn-card" id="turn-{n}" data-turn="{n}" data-part="cli">',
            f'<article class="turn-card" id="turn-{n}" data-turn="{n}" data-part="web">'
        )

    # Link Part II (Turn 28) and Part III (Turn 46) next arrows seamlessly
    cards_1_to_46 = cards_1_to_46.replace(
        '<a href="#turn-28" class="nav-arrow" title="Next Turn">▼</a>',
        '<a href="#turn-29" class="nav-arrow" title="Next Turn">▼</a>'
    )
    cards_1_to_46 = cards_1_to_46.replace(
        '<a href="#turn-46" class="nav-arrow" title="Next Turn">▼</a>',
        '<a href="#turn-47" class="nav-arrow" title="Next Turn">▼</a>'
    )

    # Generate Turn Cards 47 to 62
    cards_47_to_62_list = []
    for t in new_turns:
        num = t['num']
        t_title = t['title']
        t_summary = t['summary']
        u_raw = t['user_text']
        a_raw = t['asst_text']
        
        words = len(u_raw.split()) + len(a_raw.split())
        read_time = max(1, round(words / 200))
        resp_words = len(a_raw.split())

        u_html = markdown_to_html(u_raw)
        a_html = markdown_to_html(a_raw)

        prev_num = num - 1
        next_num = num + 1

        card_html = f"""
    <article class="turn-card" id="turn-{num}" data-turn="{num}" data-part="p4">
        <header class="turn-header">
            <div class="turn-header-left">
                <span class="turn-number-badge">Turn {num}</span>
                <span class="turn-part-badge" style="background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);">Part IV: P2 Formulary Scaling</span>
                <h3 class="turn-title">{html.escape(t_title)}</h3>
            </div>
            <div class="turn-header-right">
                <span class="stat-pill">{words:,} words (~{read_time} min read)</span>
                <button class="action-btn" onclick="copyTurnText({num})" title="Copy this turn">📋 Copy</button>
                <a href="#turn-{prev_num}" class="nav-arrow" title="Previous Turn">▲</a>
                <a href="#turn-{next_num}" class="nav-arrow" title="Next Turn">▼</a>
            </div>
        </header>

        <section class="user-bubble">
            <div class="bubble-header">
                <div class="avatar user-avatar">👤</div>
                <div class="bubble-meta">
                    <span class="bubble-sender">User</span>
                    <span class="bubble-sub">Master Viva Prep &amp; System Scaling</span>
                </div>
            </div>
            <div class="bubble-body">
                {u_html}
            </div>
        </section>

        <section class="assistant-section">
            <details class="response-details" id="details-{num}">
                <summary class="response-summary">
                    <div class="summary-left">
                        <span class="summary-toggle-icon">▸</span>
                        <span class="asst-label">Antigravity Assistant</span>
                        <span class="summary-snippet-text">{html.escape(t_summary)}</span>
                    </div>
                    <div class="summary-right">
                        <span class="badge-expand">Click to Expand / Collapse</span>
                    </div>
                </summary>
                <div class="response-body prose" id="resp-content-{num}">
                    <div class="response-actions-bar">
                        <span class="response-stat-tag">Generated by DeepMind Antigravity ({resp_words:,} words)</span>
                        <button class="copy-resp-btn" onclick="copyResponseText({num})">📋 Copy Full Response</button>
                    </div>
                    {a_html}
                </div>
            </details>
        </section>
    </article>
        """
        cards_47_to_62_list.append(card_html.strip())

    cards_47_to_62_html = "\n\n".join(cards_47_to_62_list)

    # Extract Cards 47 to 60 from existing HTML and renumber to 63 to 76
    cards_63_to_76_list = []
    for old_n in range(47, 61):
        pos_old = html_text.find(f'<article class="turn-card" id="turn-{old_n}"')
        if pos_old == -1:
            raise ValueError(f"Could not find card {old_n} in transcript.html")
        pos_old_end = html_text.find('</article>', pos_old) + len('</article>')
        card_content = html_text[pos_old:pos_old_end]

        new_n = old_n + 16
        old_prev = old_n - 1
        old_next = old_n + 1
        new_prev = new_n - 1
        new_next = new_n + 1

        c = card_content
        c = c.replace(f'id="turn-{old_n}"', f'id="turn-{new_n}"')
        c = c.replace(f'data-turn="{old_n}"', f'data-turn="{new_n}"')
        c = c.replace('data-part="p4"', 'data-part="p5"')
        c = c.replace(f'<span class="turn-number-badge">Turn {old_n}</span>', f'<span class="turn-number-badge">Turn {new_n}</span>')
        c = c.replace('Part IV: Project 2 UI & Compiler', 'Part V: Project 2 UI & Compiler')
        c = c.replace(f'onclick="copyTurnText({old_n})"', f'onclick="copyTurnText({new_n})"')
        c = c.replace(f'onclick="copyResponseText({old_n})"', f'onclick="copyResponseText({new_n})"')
        c = c.replace(f'id="details-{old_n}"', f'id="details-{new_n}"')
        c = c.replace(f'id="resp-content-{old_n}"', f'id="resp-content-{new_n}"')
        c = c.replace(f'href="#turn-{old_prev}"', f'href="#turn-{new_prev}"')
        if old_n == 60:
            c = c.replace(f'href="#turn-60"', f'href="#turn-76"')
        else:
            c = c.replace(f'href="#turn-{old_next}"', f'href="#turn-{new_next}"')

        cards_63_to_76_list.append(c.strip())

    cards_63_to_76_html = "\n\n".join(cards_63_to_76_list)

    # Extract the footer and scripts
    pos_c60 = html_text.find('id="turn-60"')
    pos_c60_end = html_text.find('</article>', pos_c60) + len('</article>')
    pos_html_end = html_text.find('</html>') + len('</html>')
    footer_and_script = html_text[pos_c60_end:pos_html_end]

    full_html = pre_sidebar + sidebar_html + middle_html + "\n\n" + cards_1_to_46 + "\n\n" + cards_47_to_62_html + "\n\n" + cards_63_to_76_html + "\n\n" + footer_and_script + "\n"
    return full_html

def main():
    print("=== Step 1: Extracting Turns 47-62 from Transcript Log ===")
    turns = extract_turns()
    print(f"Successfully extracted {len(turns)} turns.")

    print("\n=== Step 2: Generating Updated Markdown (76 Turns) ===")
    updated_md = generate_updated_markdown(turns)
    
    # Validation of MD
    found_turns = re.findall(r'^## Turn (\d+): (.*?)(?:\{#(.*?)\})?$', updated_md, re.MULTILINE)
    md_nums = [int(t[0]) for t in found_turns]
    assert md_nums == list(range(1, 77)), f"MD turn numbers mismatch: {len(md_nums)} turns found"
    print(f"Verified Markdown: exactly 76 turns sequentially (1..76).")

    all_titles = [t[1].strip() for t in found_turns]

    print("\n=== Step 3: Generating Updated HTML (76 Turns) ===")
    updated_html = generate_updated_html(turns, all_titles)

    # Validation of HTML
    card_ids = re.findall(r'<article class="turn-card" id="turn-(\d+)" data-turn="(\d+)" data-part="([^"]+)"', updated_html)
    html_nums = [int(c[0]) for c in card_ids]
    assert html_nums == list(range(1, 77)), f"HTML turn cards mismatch: {len(html_nums)} cards found"

    sidebar_items = re.findall(r'<a href="#turn-(\d+)" class="sidebar-item" id="nav-item-(\d+)" data-turn="(\d+)"', updated_html)
    sidebar_nums = [int(s[0]) for s in sidebar_items]
    assert sidebar_nums == list(range(1, 77)), f"HTML sidebar items mismatch: {len(sidebar_nums)} items found"

    assert "</html>" in updated_html, "HTML missing closing tag"
    after_html = updated_html[updated_html.find('</html>') + len('</html>'):].strip()
    assert len(after_html) == 0, f"Unclean HTML tail: {after_html[:100]}"
    print("Verified HTML: exactly 76 turn cards, 76 sidebar items, clean HTML tail.")

    # Write Master files
    print("\n=== Step 4: Writing Master Files in DSAI_Projects ===")
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(updated_md)
    print(f"Written: {MD_PATH} ({len(updated_md):,} chars)")

    with open(HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(updated_html)
    print(f"Written: {HTML_PATH} ({len(updated_html):,} chars)")

    # Propagate to subdirectories
    subdirs = [
        os.path.join(BASE_DIR, "Project1_Energy_Grid_Forecasting"),
        os.path.join(BASE_DIR, "Project2_Clinical_Shorthand_FHIR_Compiler"),
        os.path.join(BASE_DIR, "Textbook")
    ]

    print("\n=== Step 5: Propagating to Subdirectories ===")
    for sdir in subdirs:
        dest_md = os.path.join(sdir, "transcript.md")
        dest_html = os.path.join(sdir, "transcript.html")
        shutil.copyfile(MD_PATH, dest_md)
        shutil.copyfile(HTML_PATH, dest_html)
        print(f"Copied to: {sdir}")

    print("\n=== Transcript Generation & Sync Successfully Completed! ===")

if __name__ == '__main__':
    main()
