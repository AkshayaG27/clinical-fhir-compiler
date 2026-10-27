import sys, os, re, html, shutil

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = r"C:\Users\aksha\DSAI_Projects"
SRC_MD = os.path.join(BASE_DIR, "transcript_project2.md")
SRC_HTML = os.path.join(BASE_DIR, "transcript_project2.html")

with open(SRC_MD, 'r', encoding='utf-8') as f:
    md_text = f.read()

with open(SRC_HTML, 'r', encoding='utf-8') as f:
    html_text = f.read()

# ==============================================================================
# 1. RENUMBER MARKDOWN (Turns 29..76 -> Turns 1..48)
# ==============================================================================

# Extract title mapping from original MD
turn_titles = {}
for m in re.finditer(r'^## Turn (\d+): (.*?)(?:\{#(.*?)\})?$', md_text, re.MULTILINE):
    old_n = int(m.group(1))
    turn_titles[old_n] = m.group(2).strip()

print(f"Loaded {len(turn_titles)} turn titles from MD.")

# Construct new Table of Contents
p1_toc_lines = ["### Part I: Project 2 Core Architecture & Scratch Transformer (Turns 1–18)"]
for old_n in range(29, 47):
    new_n = old_n - 28
    p1_toc_lines.append(f"* [Turn {new_n}: {turn_titles[old_n]}](#turn-{new_n})")

p2_toc_lines = ["### Part II: Project 2 Formulary Scaling (50 to 384 NLEM Drugs), Training Dynamics & Dual Dashboards (Turns 19–34)"]
for old_n in range(47, 63):
    new_n = old_n - 28
    p2_toc_lines.append(f"* [Turn {new_n}: {turn_titles[old_n]}](#turn-{new_n})")

p3_toc_lines = ["### Part III: Project 2 Enterprise UI/UX, Multi-Line Compiler & Cross-Agent Tools (Turns 35–48)"]
for old_n in range(63, 77):
    new_n = old_n - 28
    p3_toc_lines.append(f"* [Turn {new_n}: {turn_titles[old_n]}](#turn-{new_n})")

new_toc = "\n".join(p1_toc_lines) + "\n\n" + "\n".join(p2_toc_lines) + "\n\n" + "\n".join(p3_toc_lines)

# Split MD into front matter and body
pos_first_turn = md_text.find('## Turn 29:')
pos_toc_start = md_text.find('## Table of Contents')
pos_toc_content_start = md_text.find('### Part III:', pos_toc_start)
pos_toc_end = md_text.find('\n---\n\n# Part III:', pos_toc_start)
if pos_toc_end == -1:
    pos_toc_end = md_text.rfind('---', pos_toc_start, pos_first_turn)

new_header = """# UMC301 Project 2: Clinical Shorthand-to-FHIR Compiler: Discussion Transcript
**Course:** Applied Data Science and Artificial Intelligence (UMC301)  
**Institution:** Indian Institute of Science (IISc), Computational and Data Sciences (CDS)  
**Instructor:** Prof. Deepak Subramani  
**Student:** Akshaya  
**Total Turns:** 48 Turns (Part I: Turns 1–18 Core Architecture | Part II: Turns 19–34 Formulary Scaling & Training | Part III: Turns 35–48 Enterprise UI/UX & Multi-Line Compiler)  
**Interactive Standalone HTML:** [`project2_transcript.html`](./project2_transcript.html)  

---

## Table of Contents

"""

body_text = md_text[pos_first_turn:].strip()

# Update Part headers in body
# In original MD:
# # Part III: Project 2 – Edge Clinical Shorthand-to-FHIR Compiler (Sprint 2) -> Part I
# # Part IV: Project 2 Formulary Scaling ... -> Part II
# # Part V: Project 2 Enterprise UI/UX ... -> Part III
# Note: before ## Turn 29, the header might have been in the slice before pos_first_turn
body_header_part1 = "# Part I: Project 2 – Edge Clinical Shorthand-to-FHIR Compiler (Sprint 2)\n\n---\n\n"

# Replace Part IV and Part V headings in body
body_text = re.sub(
    r'# Part IV: Project 2 Formulary Scaling [^\n]+',
    r'# Part II: Project 2 Formulary Scaling (50 to 384 NLEM Drugs), Training Dynamics & Dual Dashboards (Turns 19–34)',
    body_text
)
body_text = re.sub(
    r'# Part V: Project 2 Enterprise UI/UX [^\n]+',
    r'# Part III: Project 2 Enterprise UI/UX, Multi-Line Compiler & Cross-Agent Tools (Turns 35–48)',
    body_text
)

# Renumber turns from 29..76 down to 1..48 using a single-pass regex
def replace_turn_heading(m):
    old_n = int(m.group(1))
    if 29 <= old_n <= 76:
        new_n = old_n - 28
        return f"## Turn {new_n}: {m.group(2)}{{#turn-{new_n}}}"
    return m.group(0)

body_text = re.sub(
    r'^## Turn (\d+): (.*?)\{#turn-\1\}$',
    replace_turn_heading,
    body_text,
    flags=re.MULTILINE
)

# Replace Phase lines
body_text = re.sub(r'Phase:\*\* Part III:', r'Phase:** Part I:', body_text)
body_text = re.sub(r'Phase:\*\* Part IV:', r'Phase:** Part II:', body_text)
body_text = re.sub(r'Phase:\*\* Part V:', r'Phase:** Part III:', body_text)

new_md = new_header + new_toc + "\n\n---\n\n" + body_header_part1 + body_text + "\n"

# Verification of MD
new_md_turns = [int(m) for m in re.findall(r'^## Turn (\d+):', new_md, re.MULTILINE)]
new_toc_turns = [int(m) for m in re.findall(r'^\* \[Turn (\d+):', new_md, re.MULTILINE)]

print(f"New MD turns count: {len(new_md_turns)}, range: {min(new_md_turns)}..{max(new_md_turns)}")
print(f"New TOC turns count: {len(new_toc_turns)}, range: {min(new_toc_turns)}..{max(new_toc_turns)}")
assert new_md_turns == list(range(1, 49)), "New MD turns do not match 1..48"
assert new_toc_turns == list(range(1, 49)), "New TOC turns do not match 1..48"

# ==============================================================================
# 2. RENUMBER HTML (Turns 29..76 -> Turns 1..48)
# ==============================================================================

pos_sidebar = html_text.find('<aside class="sidebar"')
pre_sidebar = html_text[:pos_sidebar]

# Build new Sidebar
sidebar_lines = [
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
    '    <nav class="sidebar-scroll" id="sidebarList">'
]

sidebar_sections = [
    ("PART I: CORE ARCHITECTURE &amp; SPRINT 2", 1, 18, 29, "P2"),
    ("PART II: FORMULARY SCALING &amp; TRAINING", 19, 34, 47, "SCALE"),
    ("PART III: ENTERPRISE UI &amp; MULTI-LINE COMPILER", 35, 48, 63, "UI")
]

for sec_title, start_new, end_new, start_old, badge_tag in sidebar_sections:
    sidebar_lines.append(f'        <div class="sidebar-section-title">{sec_title}</div>')
    for new_n in range(start_new, end_new + 1):
        old_n = new_n + 28
        t_title = turn_titles[old_n]
        sidebar_lines.append(f'        <a href="#turn-{new_n}" class="sidebar-item" id="nav-item-{new_n}" data-turn="{new_n}">')
        sidebar_lines.append(f'            <span class="nav-turn-num">{new_n}</span>')
        sidebar_lines.append(f'            <div class="nav-info">')
        sidebar_lines.append(f'                <span class="nav-turn-title">{html.escape(t_title)}</span>')
        sidebar_lines.append(f'                <span class="nav-turn-tag">{badge_tag}</span>')
        sidebar_lines.append(f'            </div>')
        sidebar_lines.append(f'        </a>')

sidebar_lines.append('    </nav>')
sidebar_lines.append('</aside>')
sidebar_html = "\n".join(sidebar_lines)

middle_html = """
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
                        <div class="stat-label">Part I: Core Architecture</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">16</div>
                        <div class="stat-label">Part II: Formulary Scaling</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-num">14</div>
                        <div class="stat-label">Part III: Enterprise UI &amp; Compiler</div>
                    </div>
                </div>
            </section>

            <!-- Filter Bar -->
            <div class="filter-bar">
                <button class="filter-pill active" onclick="filterCards('all', this)">All Turns (48)</button>
                <button class="filter-pill" onclick="filterCards('p1', this)">Part I: Core Architecture (18)</button>
                <button class="filter-pill" onclick="filterCards('p2', this)">Part II: Formulary Scaling (16)</button>
                <button class="filter-pill" onclick="filterCards('p3', this)">Part III: Enterprise UI &amp; Multi-Line (14)</button>
            </div>

            <!-- Turn Cards List -->
            <div id="turnsList">
"""

# Extract all cards from old HTML
old_cards = {}
for old_n in range(29, 77):
    pos_c = html_text.find(f'<article class="turn-card" id="turn-{old_n}"')
    if pos_c == -1:
        raise ValueError(f"Card {old_n} not found in HTML")
    pos_c_end = html_text.find('</article>', pos_c) + len('</article>')
    old_cards[old_n] = html_text[pos_c:pos_c_end]

# Transform cards into new turn numbers 1..48
new_cards = []
for old_n in range(29, 77):
    new_n = old_n - 28
    c = old_cards[old_n]

    # Assign Part identifier & Badge
    if new_n <= 18:
        part_cls = "p1"
        part_badge = '<span class="turn-part-badge" style="background-color: rgba(99, 102, 241, 0.2); color: var(--accent-primary); border: 1px solid rgba(99, 102, 241, 0.4);">Part I: Core Architecture</span>'
    elif new_n <= 34:
        part_cls = "p2"
        part_badge = '<span class="turn-part-badge" style="background-color: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.4);">Part II: Formulary Scaling</span>'
    else:
        part_cls = "p3"
        part_badge = '<span class="turn-part-badge" style="background-color: rgba(20, 184, 166, 0.2); color: var(--accent-teal); border: 1px solid rgba(20, 184, 166, 0.4);">Part III: Enterprise UI & Compiler</span>'

    # Replace article tag
    c = re.sub(
        r'<article class="turn-card" id="turn-\d+" data-turn="\d+" data-part="[^"]+"',
        f'<article class="turn-card" id="turn-{new_n}" data-turn="{new_n}" data-part="{part_cls}"',
        c
    )

    # Replace Turn number badge
    c = re.sub(
        r'<span class="turn-number-badge">Turn \d+</span>',
        f'<span class="turn-number-badge">Turn {new_n}</span>',
        c
    )

    # Replace Part badge
    c = re.sub(
        r'<span class="turn-part-badge"[^>]*>.*?</span>',
        part_badge,
        c
    )

    # Replace copy button
    c = re.sub(
        r'onclick="copyTurnText\(\d+\)"',
        f'onclick="copyTurnText({new_n})"',
        c
    )

    # Replace response copy button if present
    c = re.sub(
        r'onclick="copyResponseText\(\d+\)"',
        f'onclick="copyResponseText({new_n})"',
        c
    )

    # Replace details ID
    c = re.sub(
        r'id="details-\d+"',
        f'id="details-{new_n}"',
        c
    )
    c = re.sub(
        r'id="details-turn-\d+"',
        f'id="details-turn-{new_n}"',
        c
    )

    # Replace resp-content ID
    c = re.sub(
        r'id="resp-content-\d+"',
        f'id="resp-content-{new_n}"',
        c
    )

    # Navigation arrows: prev and next
    prev_n = 1 if new_n == 1 else new_n - 1
    next_n = 48 if new_n == 48 else new_n + 1

    # Replace arrow hrefs in header
    # Find turn-header-right
    hdr_right_match = re.search(r'<div class="turn-header-right">.*?</div>', c, re.DOTALL)
    if hdr_right_match:
        old_hdr_right = hdr_right_match.group(0)
        # replace arrows inside this block
        new_hdr_right = re.sub(
            r'<a href="#turn-\d+" class="nav-arrow" title="Previous Turn">▲</a>',
            f'<a href="#turn-{prev_n}" class="nav-arrow" title="Previous Turn">▲</a>',
            old_hdr_right
        )
        new_hdr_right = re.sub(
            r'<a href="#turn-\d+" class="nav-arrow" title="Next Turn">▼</a>',
            f'<a href="#turn-{next_n}" class="nav-arrow" title="Next Turn">▼</a>',
            new_hdr_right
        )
        c = c[:hdr_right_match.start()] + new_hdr_right + c[hdr_right_match.end():]

    new_cards.append(c.strip())

# Extract footer and scripts from original HTML
pos_last_card = html_text.find('id="turn-76"')
pos_last_card_end = html_text.find('</article>', pos_last_card) + len('</article>')
pos_html_end = html_text.find('</html>') + len('</html>')
footer_and_script = html_text[pos_last_card_end:pos_html_end]

new_html = pre_sidebar + sidebar_html + middle_html + "\n\n" + "\n\n".join(new_cards) + "\n\n" + footer_and_script + "\n"

# Verification of HTML
html_card_ids = [int(m) for m in re.findall(r'<article class="turn-card" id="turn-(\d+)"', new_html)]
html_sidebar_ids = [int(m) for m in re.findall(r'id="nav-item-(\d+)"', new_html)]

print(f"New HTML cards count: {len(html_card_ids)}, range: {min(html_card_ids)}..{max(html_card_ids)}")
print(f"New HTML sidebar items count: {len(html_sidebar_ids)}, range: {min(html_sidebar_ids)}..{max(html_sidebar_ids)}")

assert html_card_ids == list(range(1, 49)), "New HTML cards mismatch"
assert html_sidebar_ids == list(range(1, 49)), "New HTML sidebar items mismatch"

# Verify arrow navigation continuity in HTML
for new_n in range(1, 49):
    pos_c = new_html.find(f'<article class="turn-card" id="turn-{new_n}"')
    pos_c_end = new_html.find('</article>', pos_c)
    card_str = new_html[pos_c:pos_c_end]
    arrows = re.findall(r'<a href="#turn-(\d+)" class="nav-arrow"', card_str)
    assert len(arrows) == 2, f"Card {new_n} arrows count != 2"
    p_n, n_n = int(arrows[0]), int(arrows[1])
    exp_p = 1 if new_n == 1 else new_n - 1
    exp_n = 48 if new_n == 48 else new_n + 1
    assert p_n == exp_p, f"Card {new_n} prev arrow got {p_n}, exp {exp_p}"
    assert n_n == exp_n, f"Card {new_n} next arrow got {n_n}, exp {exp_n}"

print("✓ All 48 HTML cards verified with continuous 1..48 navigation arrows!")

# ==============================================================================
# 3. SAVE TO ALL TARGET LOCATIONS
# ==============================================================================

# Root locations
with open(os.path.join(BASE_DIR, "transcript_project2.md"), 'w', encoding='utf-8') as f:
    f.write(new_md)
with open(os.path.join(BASE_DIR, "project2_transcript.md"), 'w', encoding='utf-8') as f:
    f.write(new_md)

with open(os.path.join(BASE_DIR, "transcript_project2.html"), 'w', encoding='utf-8') as f:
    f.write(new_html)
with open(os.path.join(BASE_DIR, "project2_transcript.html"), 'w', encoding='utf-8') as f:
    f.write(new_html)

print("Saved transcript_project2.html/md and project2_transcript.html/md in root.")

# Project 2 directory locations
p2_dir = os.path.join(BASE_DIR, "Project2_Clinical_Shorthand_FHIR_Compiler")
with open(os.path.join(p2_dir, "transcript.md"), 'w', encoding='utf-8') as f:
    f.write(new_md)
with open(os.path.join(p2_dir, "transcript.html"), 'w', encoding='utf-8') as f:
    f.write(new_html)

print(f"Saved transcript.html/md in {p2_dir}.")
print("=== Renumbering complete and fully verified! ===")
