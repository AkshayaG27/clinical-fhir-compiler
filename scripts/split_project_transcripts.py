import sys, os, re, html, shutil

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\aksha\DSAI_Projects"
MASTER_MD = os.path.join(BASE_DIR, "transcript.md")
MASTER_HTML = os.path.join(BASE_DIR, "transcript.html")

with open(MASTER_MD, 'r', encoding='utf-8') as f:
    master_md_text = f.read()

with open(MASTER_HTML, 'r', encoding='utf-8') as f:
    master_html_text = f.read()

# ==============================================================================
# 1. SPLIT MARKDOWN TRANSCRIPT
# ==============================================================================

# Find boundaries in Master MD
pos_t28 = master_md_text.find('## Turn 28:')
pos_t29 = master_md_text.find('## Turn 29:')

# Find TOC boundaries in Master MD
toc_part1_pos = master_md_text.find('### Part I:')
toc_part3_pos = master_md_text.find('### Part III:')
body_t1_pos = master_md_text.find('## Turn 1:')
body_t29_pos = master_md_text.find('## Turn 29:')

# Part 1 TOC & Body
p1_toc = master_md_text[toc_part1_pos:toc_part3_pos].strip()
p1_body = master_md_text[body_t1_pos:body_t29_pos].strip()

# Clean any trailing "---" or Part III headers from p1_body
if "# Part III:" in p1_body:
    p1_body = p1_body[:p1_body.find('# Part III:')].strip()
p1_body = re.sub(r'---\s*$', '', p1_body).strip()

p1_md_header = """# UMC301 Project 1: Energy Grid Forecasting & Portfolio Strategy: Discussion Transcript
**Course:** Applied Data Science and Artificial Intelligence (UMC301)  
**Institution:** Indian Institute of Science (IISc), Computational and Data Sciences (CDS)  
**Instructor:** Prof. Deepak Subramani  
**Student:** Akshaya  
**Total Turns:** 28 Turns (Part I: Turns 1–10 CLI Foundations | Part II: Turns 11–28 Web/IDE Strategy & Multi-Project Portfolio)  
**Interactive Standalone HTML:** [`transcript.html`](./transcript.html)  

---

## Table of Contents

"""

project1_md = p1_md_header + p1_toc + "\n\n---\n\n" + p1_body + "\n"

# Part 2 (Project 2: Parts 3, 4, 5) TOC & Body
toc_end_pos = master_md_text.find('\n---\n\n## Turn 1:')
if toc_end_pos == -1:
    toc_end_pos = master_md_text.find('\n---\n\n# Part I')
if toc_end_pos == -1:
    # find first ## Turn 1:
    toc_end_pos = master_md_text.find('## Turn 1:')
    # walk back to previous ---
    toc_end_pos = master_md_text.rfind('---', 0, toc_end_pos)

p2_toc = master_md_text[toc_part3_pos:toc_end_pos].strip()
p2_body = master_md_text[body_t29_pos:].strip()

# Add Part III header to p2_body
p2_body = "# Part III: Project 2 – Edge Clinical Shorthand-to-FHIR Compiler (Sprint 2)\n\n---\n\n" + p2_body

p2_md_header = """# UMC301 Project 2: Clinical Shorthand-to-FHIR Compiler: Discussion Transcript
**Course:** Applied Data Science and Artificial Intelligence (UMC301)  
**Institution:** Indian Institute of Science (IISc), Computational and Data Sciences (CDS)  
**Instructor:** Prof. Deepak Subramani  
**Student:** Akshaya  
**Total Turns:** 48 Turns (Part III: Turns 29–46 Core Architecture | Part IV: Turns 47–62 Formulary Scaling & Training | Part V: Turns 63–76 Enterprise UI/UX & Multi-Line Compiler)  
**Interactive Standalone HTML:** [`transcript.html`](./transcript.html)  

---

## Table of Contents

"""

project2_md = p2_md_header + p2_toc + "\n\n---\n\n" + p2_body + "\n"

# Verify MD turns
p1_turns = [int(m) for m in re.findall(r'^## Turn (\d+):', project1_md, re.MULTILINE)]
p2_turns = [int(m) for m in re.findall(r'^## Turn (\d+):', project2_md, re.MULTILINE)]

print(f"Project 1 MD turns count: {len(p1_turns)}, range: {min(p1_turns)}..{max(p1_turns)}")
print(f"Project 2 MD turns count: {len(p2_turns)}, range: {min(p2_turns)}..{max(p2_turns)}")
assert p1_turns == list(range(1, 29)), "Project 1 MD turns not 1..28"
assert p2_turns == list(range(29, 77)), "Project 2 MD turns not 29..76"

# ==============================================================================
# 2. EXTRACT TITLES FOR SIDEBAR
# ==============================================================================
found_titles = {}
for m in re.finditer(r'^## Turn (\d+): (.*?)(?:\{#(.*?)\})?$', master_md_text, re.MULTILINE):
    found_titles[int(m.group(1))] = m.group(2).strip()

# Extract common HTML template parts
pos_sidebar = master_html_text.find('<aside class="sidebar"')
pre_sidebar = master_html_text[:pos_sidebar]

pos_c60 = master_html_text.find('id="turn-76"')
pos_c60_end = master_html_text.find('</article>', pos_c60) + len('</article>')
pos_html_end = master_html_text.find('</html>') + len('</html>')
footer_and_script = master_html_text[pos_c60_end:pos_html_end]

# Extract card HTMLs
cards = {}
for n in range(1, 77):
    pos_cn = master_html_text.find(f'<article class="turn-card" id="turn-{n}"')
    if pos_cn == -1:
        raise ValueError(f"Card {n} not found")
    pos_cn_end = master_html_text.find('</article>', pos_cn) + len('</article>')
    cards[n] = master_html_text[pos_cn:pos_cn_end]

# ==============================================================================
# 3. GENERATE PROJECT 1 HTML
# ==============================================================================

# Project 1 Sidebar
p1_sidebar_lines = [
    '<aside class="sidebar" id="sidebar">',
    '    <div class="sidebar-header">',
    '        <div class="sidebar-title">',
    '            <span>📑</span> Transcript Index',
    '        </div>',
    '        <div class="sidebar-subtitle">Project 1 &amp; Strategy | 28 Turns</div>',
    '    </div>',
    '    <div class="sidebar-search-box">',
    '        <input type="text" class="sidebar-search-input" id="sidebarSearch" placeholder="Filter turns..." oninput="filterSidebar(this.value)">',
    '    </div>',
    '    <nav class="sidebar-scroll" id="sidebarList">',
    '        <div class="sidebar-section-title">PART I: CLI FOUNDATIONS</div>'
]
for n in range(1, 11):
    t_title = found_titles[n]
    p1_sidebar_lines.append(f'        <a href="#turn-{n}" class="sidebar-item" id="nav-item-{n}" data-turn="{n}">')
    p1_sidebar_lines.append(f'            <span class="nav-turn-num">{n}</span>')
    p1_sidebar_lines.append(f'            <div class="nav-info">')
    p1_sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
    p1_sidebar_lines.append(f'                <span class="nav-turn-tag">CLI</span>')
    p1_sidebar_lines.append(f'            </div>')
    p1_sidebar_lines.append(f'        </a>')

p1_sidebar_lines.append('        <div class="sidebar-section-title">PART II: WEB / IDE PORTFOLIO</div>')
for n in range(11, 29):
    t_title = found_titles[n]
    p1_sidebar_lines.append(f'        <a href="#turn-{n}" class="sidebar-item" id="nav-item-{n}" data-turn="{n}">')
    p1_sidebar_lines.append(f'            <span class="nav-turn-num">{n}</span>')
    p1_sidebar_lines.append(f'            <div class="nav-info">')
    p1_sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
    p1_sidebar_lines.append(f'                <span class="nav-turn-tag">IDE</span>')
    p1_sidebar_lines.append(f'            </div>')
    p1_sidebar_lines.append(f'        </a>')

p1_sidebar_lines.append('    </nav>')
p1_sidebar_lines.append('</aside>')
p1_sidebar_html = "\n".join(p1_sidebar_lines)

p1_middle_html = """
    <!-- Main Container -->
    <div class="main-wrapper">
        <!-- Sticky Navbar -->
        <header class="top-navbar">
            <div class="top-nav-left">
                <button class="btn mobile-menu-btn" onclick="toggleSidebar()">☰ Menu</button>
                <h1 class="doc-title">Project 1 Discussion Transcript</h1>
                <span class="doc-tag">UMC301 Energy Grid & Strategy</span>
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
                <h2 class="hero-title">IISc UMC301 Project 1: Energy Grid Forecasting & Portfolio Strategy</h2>
                <p class="hero-desc">
                    Comprehensive chronological record of strategic architecture, PACF feature engineering, non-parametric hypothesis testing, quantile shrinkage calibration, and multi-pillar portfolio roadmap.
                </p>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-num">28</div>
                        <div class="stat-label">Total Conversation Turns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">10</div>
                        <div class="stat-label">Part I: CLI Session Turns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">18</div>
                        <div class="stat-label">Part II: Web/IDE Strategy Turns</div>
                    </div>
                </div>
            </section>

            <!-- Filter Bar -->
            <div class="filter-bar">
                <button class="filter-pill active" onclick="filterCards('all', this)">All Turns (28)</button>
                <button class="filter-pill" onclick="filterCards('cli', this)">Part I: CLI Session (10)</button>
                <button class="filter-pill" onclick="filterCards('web', this)">Part II: Web/IDE (18)</button>
            </div>

            <!-- Turn Cards List -->
            <div id="turnsList">
"""

# Adjust Card 28 next arrow to link to #turn-28 in Project 1 page
p1_cards = []
for n in range(1, 29):
    c = cards[n]
    if n == 28:
        c = c.replace('<a href="#turn-29" class="nav-arrow" title="Next Turn">▼</a>', '<a href="#turn-28" class="nav-arrow" title="Next Turn">▼</a>')
    p1_cards.append(c)

project1_html = pre_sidebar + p1_sidebar_html + p1_middle_html + "\n\n" + "\n\n".join(p1_cards) + "\n\n" + footer_and_script + "\n"

# ==============================================================================
# 4. GENERATE PROJECT 2 HTML
# ==============================================================================

# Project 2 Sidebar
p2_sidebar_lines = [
    '<aside class="sidebar" id="sidebar">',
    '    <div class="sidebar-header">',
    '        <div class="sidebar-title">',
    '            <span>📑</span> Transcript Index',
    '        </div>',
    '        <div class="sidebar-subtitle">Project 2: Clinical Compiler | 48 Turns</div>',
    '    </div>',
    '    <div class="sidebar-search-box">',
    '        <input type="text" class="sidebar-search-input" id="sidebarSearch" placeholder="Filter turns..." oninput="filterSidebar(this.value)">',
    '    </div>',
    '    <nav class="sidebar-scroll" id="sidebarList">',
    '        <div class="sidebar-section-title">PART III: CORE ARCHITECTURE &amp; SPRINT 2</div>'
]
for n in range(29, 47):
    t_title = found_titles[n]
    p2_sidebar_lines.append(f'        <a href="#turn-{n}" class="sidebar-item" id="nav-item-{n}" data-turn="{n}">')
    p2_sidebar_lines.append(f'            <span class="nav-turn-num">{n}</span>')
    p2_sidebar_lines.append(f'            <div class="nav-info">')
    p2_sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
    p2_sidebar_lines.append(f'                <span class="nav-turn-tag">P2</span>')
    p2_sidebar_lines.append(f'            </div>')
    p2_sidebar_lines.append(f'        </a>')

p2_sidebar_lines.append('        <div class="sidebar-section-title">PART IV: FORMULARY SCALING &amp; TRAINING</div>')
for n in range(47, 63):
    t_title = found_titles[n]
    p2_sidebar_lines.append(f'        <a href="#turn-{n}" class="sidebar-item" id="nav-item-{n}" data-turn="{n}">')
    p2_sidebar_lines.append(f'            <span class="nav-turn-num">{n}</span>')
    p2_sidebar_lines.append(f'            <div class="nav-info">')
    p2_sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
    p2_sidebar_lines.append(f'                <span class="nav-turn-tag">SCALE</span>')
    p2_sidebar_lines.append(f'            </div>')
    p2_sidebar_lines.append(f'        </a>')

p2_sidebar_lines.append('        <div class="sidebar-section-title">PART V: ENTERPRISE UI &amp; MULTI-LINE COMPILER</div>')
for n in range(63, 77):
    t_title = found_titles[n]
    p2_sidebar_lines.append(f'        <a href="#turn-{n}" class="sidebar-item" id="nav-item-{n}" data-turn="{n}">')
    p2_sidebar_lines.append(f'            <span class="nav-turn-num">{n}</span>')
    p2_sidebar_lines.append(f'            <div class="nav-info">')
    p2_sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
    p2_sidebar_lines.append(f'                <span class="nav-turn-tag">UI</span>')
    p2_sidebar_lines.append(f'            </div>')
    p2_sidebar_lines.append(f'        </a>')

p2_sidebar_lines.append('    </nav>')
p2_sidebar_lines.append('</aside>')
p2_sidebar_html = "\n".join(p2_sidebar_lines)

p2_middle_html = """
    <!-- Main Container -->
    <div class="main-wrapper">
        <!-- Sticky Navbar -->
        <header class="top-navbar">
            <div class="top-nav-left">
                <button class="btn mobile-menu-btn" onclick="toggleSidebar()">☰ Menu</button>
                <h1 class="doc-title">Project 2 Discussion Transcript</h1>
                <span class="doc-tag">UMC301 Clinical FHIR Compiler</span>
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
                <h2 class="hero-title">IISc UMC301 Project 2: Edge Clinical Shorthand-to-FHIR Compiler</h2>
                <p class="hero-desc">
                    Comprehensive chronological record of architectural design, scratch causal Transformer derivation, RoPE relative distance invariance, deterministic DFA logit masking, NLEM 2022 scaling, and multi-line statement decomposition.
                </p>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-num">48</div>
                        <div class="stat-label">Total Conversation Turns</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">18</div>
                        <div class="stat-label">Part III: Core Architecture</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">16</div>
                        <div class="stat-label">Part IV: Formulary Scaling</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">14</div>
                        <div class="stat-label">Part V: Enterprise UI & Compiler</div>
                    </div>
                </div>
            </section>

            <!-- Filter Bar -->
            <div class="filter-bar">
                <button class="filter-pill active" onclick="filterCards('all', this)">All P2 Turns (48)</button>
                <button class="filter-pill" onclick="filterCards('p2', this)">Part III: Core Architecture (18)</button>
                <button class="filter-pill" onclick="filterCards('p4', this)">Part IV: Formulary Scaling (16)</button>
                <button class="filter-pill" onclick="filterCards('p5', this)">Part V: Enterprise UI & Multi-Line (14)</button>
            </div>

            <!-- Turn Cards List -->
            <div id="turnsList">
"""

# Adjust Card 29 prev arrow to link to #turn-29 in Project 2 page
p2_cards = []
for n in range(29, 77):
    c = cards[n]
    if n == 29:
        c = c.replace('<a href="#turn-28" class="nav-arrow" title="Previous Turn">▲</a>', '<a href="#turn-29" class="nav-arrow" title="Previous Turn">▲</a>')
    p2_cards.append(c)

project2_html = pre_sidebar + p2_sidebar_html + p2_middle_html + "\n\n" + "\n\n".join(p2_cards) + "\n\n" + footer_and_script + "\n"

# Verify HTML cards and sidebar items
p1_card_ids = [int(m) for m in re.findall(r'<article class="turn-card" id="turn-(\d+)"', project1_html)]
p1_sidebar_ids = [int(m) for m in re.findall(r'id="nav-item-(\d+)"', project1_html)]

p2_card_ids = [int(m) for m in re.findall(r'<article class="turn-card" id="turn-(\d+)"', project2_html)]
p2_sidebar_ids = [int(m) for m in re.findall(r'id="nav-item-(\d+)"', project2_html)]

print(f"Project 1 HTML cards: {len(p1_card_ids)}, sidebar: {len(p1_sidebar_ids)}")
print(f"Project 2 HTML cards: {len(p2_card_ids)}, sidebar: {len(p2_sidebar_ids)}")

assert p1_card_ids == list(range(1, 29)), "P1 HTML cards mismatch"
assert p1_sidebar_ids == list(range(1, 29)), "P1 HTML sidebar mismatch"

assert p2_card_ids == list(range(29, 77)), "P2 HTML cards mismatch"
assert p2_sidebar_ids == list(range(29, 77)), "P2 HTML sidebar mismatch"

# ==============================================================================
# 5. WRITE AND PROPAGATE FILES
# ==============================================================================

# Write standalone files in root
with open(os.path.join(BASE_DIR, "transcript_project1.md"), 'w', encoding='utf-8') as f:
    f.write(project1_md)
with open(os.path.join(BASE_DIR, "transcript_project1.html"), 'w', encoding='utf-8') as f:
    f.write(project1_html)

with open(os.path.join(BASE_DIR, "transcript_project2.md"), 'w', encoding='utf-8') as f:
    f.write(project2_md)
with open(os.path.join(BASE_DIR, "transcript_project2.html"), 'w', encoding='utf-8') as f:
    f.write(project2_html)

print("Written standalone files in root: transcript_project1 and transcript_project2")

# Write to Project 1 directory
p1_dir = os.path.join(BASE_DIR, "Project1_Energy_Grid_Forecasting")
with open(os.path.join(p1_dir, "transcript.md"), 'w', encoding='utf-8') as f:
    f.write(project1_md)
with open(os.path.join(p1_dir, "transcript.html"), 'w', encoding='utf-8') as f:
    f.write(project1_html)
print(f"Written dedicated transcript.md & transcript.html to {p1_dir}")

# Write to Project 2 directory
p2_dir = os.path.join(BASE_DIR, "Project2_Clinical_Shorthand_FHIR_Compiler")
with open(os.path.join(p2_dir, "transcript.md"), 'w', encoding='utf-8') as f:
    f.write(project2_md)
with open(os.path.join(p2_dir, "transcript.html"), 'w', encoding='utf-8') as f:
    f.write(project2_html)
print(f"Written dedicated transcript.md & transcript.html to {p2_dir}")

print("=== ALL TRANSCRIPTS SUCCESSFULLY SPLIT AND DEPLOYED! ===")

# Verify Arrow chains
for name, p_path, start_t, end_t in [
    ('Project 1', os.path.join(BASE_DIR, 'transcript_project1.html'), 1, 28),
    ('Project 2', os.path.join(BASE_DIR, 'transcript_project2.html'), 29, 76)
]:
    with open(p_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    card_hdrs = re.findall(r'<article class="turn-card" id="turn-(\d+)".*?<header class="turn-header">(.*?)</header>', html_content, re.DOTALL)
    for c_num, hdr in card_hdrs:
        c_n = int(c_num)
        arrows = re.findall(r'<a href="#turn-(\d+)" class="nav-arrow"', hdr)
        assert len(arrows) == 2, f"Card {c_n} in {name} does not have 2 arrows"
        p_n, n_n = int(arrows[0]), int(arrows[1])
        exp_p = start_t if c_n == start_t else c_n - 1
        exp_n = end_t if c_n == end_t else c_n + 1
        assert p_n == exp_p, f"{name} card {c_n}: prev got {p_n}, exp {exp_p}"
        assert n_n == exp_n, f"{name} card {c_n}: next got {n_n}, exp {exp_n}"
    print(f"✓ {name} arrow chain {start_t}..{end_t} is 100% verified!")
