"""Automatic validation (test oracle) for the Question/Answering tests.

Every chatbot answer is checked against the reference key points of its test
case and mapped to exactly one output class of the 3D model:

    QA-OC-01  Accurate, relevant and complete      Pass
    QA-OC-02  Nonexistent subject identified       Pass
    QA-OC-03  Accurate but incomplete              Partial Pass
    QA-OC-04  Accurate but unfocused (too long)    Partial Pass
    QA-OC-05  Inaccurate                           Fail
    QA-OC-06  Fabricated                           Fail
    QA-OC-07  Irrelevant / misses most key points  Fail

The checks are plain rules (regex, number parsing, command parsing), so the
same answer always gets the same verdict. qa_judge.py adds an optional second
opinion from a local LLM.
"""

import re
import shlex
import unicodedata

OC_VERDICT = {
    "QA-OC-01": "Pass",
    "QA-OC-02": "Pass",
    "QA-OC-03": "Partial Pass",
    "QA-OC-04": "Partial Pass",
    "QA-OC-05": "Fail",
    "QA-OC-06": "Fail",
    "QA-OC-07": "Fail",
}

NEGATION = re.compile(
    r"\b(not|no|never|without|cannot|nor|none)\b|n't\b|\bun(acknowledged|confirmed)\b")


# ---------------------------------------------------------------- helpers

def normalize(text):
    """Unify quotes, dashes, spaces and math signs that chat UIs render differently."""
    text = unicodedata.normalize("NFKC", text or "")
    for a, b in (("‘", "'"), ("’", "'"), ("“", '"'), ("”", '"'),
                 ("−", "-"), ("–", "-"), ("—", " - "), ("×", "x"),
                 ("≈", "~"), (" ", " "), (" ", " ")):
        text = text.replace(a, b)
    return text


def sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def negated(sentence, start, window=40):
    """True if a negation word sits shortly before position `start` in the sentence."""
    before = sentence[max(0, start - window):start]
    return bool(NEGATION.search(before))


def kp(name, ok, detail=""):
    return {"key_point": name, "ok": bool(ok), "detail": detail}


def classify(key_points, errors, fabrications, words, max_words, nonexistent=False):
    """Map key point hits and error flags to one output class."""
    total = len(key_points)
    hit = sum(1 for k in key_points if k["ok"])
    missing = total - hit
    if fabrications:
        oc = "QA-OC-06"
    elif errors:
        oc = "QA-OC-05"
    elif missing == 0:
        oc = "QA-OC-04" if words > max_words else ("QA-OC-02" if nonexistent else "QA-OC-01")
    elif hit == 0 or missing > total / 2:
        oc = "QA-OC-07"
    else:
        oc = "QA-OC-03"
    return oc


# ------------------------------------------------------- TC-QA-01 GetBulk

VERSION_RE = re.compile(
    r"\bsnmp\s*-?\s*v\s*([123])(?:c|u)?\b|\bsnmp\s+version\s+([123])\b|"
    r"\bversion\s+([123])c?\b|\bv([123])c?\b")
INTRO_RE = re.compile(r"introduc|added|first (appeared|defined|available)|new in|"
                      r"defined in|came with|came in|debuted|brought in")


def _versions(s):
    return {next(g for g in m.groups() if g) for m in VERSION_RE.finditer(s)}


def check_getbulk_version(answer, spec):
    t = normalize(answer).lower()
    errors = []
    named = re.search(r"get\s*-?\s*bulk", t)
    v2 = "2" in _versions(t)
    for s in sentences(t):
        m = INTRO_RE.search(s)
        if not m or not re.search(r"get\s*-?\s*bulk|\bit\b|\bthis (operation|pdu|request)", s):
            continue
        vs = _versions(s)
        if vs and "2" not in vs and not negated(s, m.start()):
            errors.append("states GetBulk was introduced in SNMPv%s" % "/".join(sorted(vs)))
    kps = [kp("Names GetBulk / GetBulkRequest", named),
           kp("Says it came with SNMPv2", v2 and not errors)]
    return kps, errors, []


# ------------------------------------------------- TC-QA-02 OID and unit

OID_RE = re.compile(r"(?<![\w.])\.?(\d+(?:\.\d+){4,})")
UNIT_OK = re.compile(r"hundredths?|1\s*/\s*100\b|0\.01\s*(s|sec|second)|centisecond|"
                     r"10\s*(ms|milli)|one[- ]hundredth|100ths?")
UNIT_WRONG = re.compile(r"(in|of|measured in|unit is|units? (is|are)|counts?|counted in)\s+"
                        r"(whole\s+)?(milliseconds|microseconds|seconds|ticks of 1 ?ms)\b")


def check_oid_unit(answer, spec):
    t = normalize(answer)
    exp = spec["oid"]
    oids = [o.lstrip(".") for o in OID_RE.findall(t)]
    right = [o for o in oids if o in (exp, exp + ".0")]
    errors = []
    if not right:
        wrong = sorted({o for o in oids if o.startswith("1.3.6.1")})
        if wrong:
            errors.append("gives OID %s, expected %s" % (", ".join(wrong), exp))
    unit_ok = bool(UNIT_OK.search(t.lower()))
    if not unit_ok and UNIT_WRONG.search(t.lower()):
        errors.append("gives the wrong unit (expected hundredths of a second)")
    kps = [kp("Exact OID %s" % exp, right, ", ".join(sorted(set(oids))) or "no OID found"),
           kp("Unit is hundredths of a second (TimeTicks)", unit_ok)]
    return kps, errors, []


# ----------------------------------------------- TC-QA-03 counter wrap math

RATE_RE = re.compile(
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*"
    r"((?i:kilo|mega|giga|k|m|g))?\s*"
    r"(bps\b|b/s\b|(?i:bits?)\s*/\s*s(?:ec(?:ond)?)?\b|(?i:bits?)\s+per\s+second)")
SCALE = {"": 1, "k": 1e3, "kilo": 1e3, "m": 1e6, "mega": 1e6, "g": 1e9, "giga": 1e9}


def expected_rate(spec):
    delta = (2 ** spec.get("bits", 32) - spec["first"]) + spec["second"]
    return delta, delta * 8, delta * 8 / spec["interval"]


def _step(num):
    """Size of the last digit the number states: 254,629.6 -> 0.1, 262 -> 1, 0.26 -> 0.01, 262,300 -> 100."""
    num = num.replace(",", "")
    if "." in num:
        return 10 ** -len(num.split(".")[1])
    return 10 ** (len(num) - len(num.rstrip("0"))) if num.strip("0") else 1


def parse_rates(text, with_step=False):
    out = []
    for num, prefix, _unit in RATE_RE.findall(text):
        try:
            value = float(num.replace(",", ""))
        except ValueError:
            continue
        scale = SCALE[(prefix or "").lower()]
        out.append((value * scale, _step(num) * scale) if with_step else value * scale)
    return out


def check_counter_wrap(answer, spec):
    t = normalize(answer)
    delta, _bits, bps = expected_rate(spec)
    tol = spec.get("tolerance", 0.01)
    rates = parse_rates(t)
    good = [r for r in rates if abs(r - bps) / bps <= tol]
    # a second result near the answer that does not match it (beyond its own rounding) means the
    # answer contradicts itself, e.g. a headline of 254,629.6 bps over working that reaches 262,306
    conflicting = [r for r, step in parse_rates(t, with_step=True)
                   if 0.5 * bps <= r <= 2 * bps and abs(r - bps) > max(tol * bps, step / 2)]
    delta_txt = re.compile(r"\b" + r"[,\s]?".join(re.findall(r"\d{1,3}", "{:,}".format(delta))) + r"\b")
    wrap = bool(re.search(r"2\s*\^\s*32|2\*\*32|\b2\n32\b|4[,\s]?294[,\s]?967[,\s]?29[56]|wrap|"
                          r"roll(ed|s)?[\s-]?over|overflow", t, re.I)) or bool(delta_txt.search(t))
    errors = []
    if rates and not good:
        errors.append("final rate %s bps is not within %d%% of %.0f bps"
                      % (", ".join("%.0f" % r for r in rates[-3:]), tol * 100, bps))
    elif good and conflicting:
        errors.append("gives conflicting results: %s bps next to %.0f bps"
                      % (", ".join("%.1f" % r for r in conflicting[:2]), bps))
    kps = [kp("Handles the counter wrap", wrap or good),
           kp("Average rate about %.0f bps (within %d%%)" % (bps, tol * 100), good,
              "rates found: " + (", ".join("%.0f" % r for r in rates) or "none"))]
    return kps, errors, []


# ---------------------------------------------------- TC-QA-04 UDP reasons

UDP_REASONS = {
    "low overhead": r"overhead|lightweight|light-weight|small(er)? header|less (bandwidth|traffic)|"
                    r"efficien|low[- ]cost|lean",
    "no connection setup": r"connectionless|connection-less|no (connection|handshake|session)|"
                           r"without (a |any )?(connection|handshake|session)|handshake|"
                           r"(connection|session) (setup|set-up|establishment)",
    "works when the network is degraded": r"congest|degraded|network (problems|issues|trouble|failures?|"
                                          r"outages?|stress)|under (load|stress)|lossy|packet loss|"
                                          r"back-?off|head-of-line|when the network is",
    "manager handles retries": r"retr(y|ies|ied)|timeouts?|time-outs?|resend|application(-| )level",
    "simple, low-resource agents": r"simplicity|simple to implement|simpler|low[- ]resource|limited "
                                   r"(memory|cpu|resources)|embedded|constrained|stateless|less memory",
    "small request/response messages": r"single (datagram|packet)|one (datagram|packet)|small "
                                       r"(messages|requests|packets)|request[- /]response",
    "scales to many devices": r"scal(e|es|able|ability)|many devices|thousands of",
}
UDP_WRONG = [r"snmp (normally |typically |usually |mostly )?(uses|runs over|runs on) tcp\b",
             r"udp (is|provides|guarantees) (reliable|guaranteed)"]


def check_udp_reasons(answer, spec):
    t = normalize(answer).lower()
    found = [name for name, pat in UDP_REASONS.items() if re.search(pat, t)]
    errors = []
    for s in sentences(t):
        for pat in UDP_WRONG:
            m = re.search(pat, s)
            if m and not negated(s, m.start()) and not NEGATION.search(m.group(0)):
                errors.append("wrong claim: '%s'" % m.group(0))
    need = spec.get("min_reasons", 2)
    kps = [kp("Reason %d" % (i + 1), len(found) > i, found[i] if len(found) > i else "")
           for i in range(need)]
    return kps, errors, []


# ------------------------------------------------- TC-QA-05 trap vs inform

ACK = r"(acknowledg\w*|confirm\w*|response pdu|repl(y|ies)|\back\b|retransmit\w*|resen[dt]\w*|retr(y|ies|ied))"
TRAP_NO_ACK = [
    r"traps?\b[^.\n]{0,80}?(not acknowledged|unacknowledged|no acknowledg\w*|without (\w+ ){0,2}(an? |any )?"
    r"(acknowledg\w*|confirm\w*|response|reply)|no (confirmation|response|reply|ack)\b|not confirmed|"
    r"fire[- ]and[- ]forget|(doesn't|does not|don't|do not|never) (get|receive|require|expect|wait)|"
    r"one-way|no way (to know|of knowing))",
    r"(unacknowledged|fire[- ]and[- ]forget|one-way)[^.\n]{0,40}?traps?\b",
]
INFORM_ACK = [r"informs?\b[^.\n]{0,100}?" + ACK, r"(acknowledged|confirmed)[^.\n]{0,30}?informs?\b"]
TRAP_INFORM_WRONG = [r"informs? (are|is) (not|never) acknowledged", r"informs? (are|is) unacknowledged",
                     r"\btraps? (are|is) acknowledged"]


CONDITION = re.compile(r"\b(if|unless|until|when|whenever|whether|in case)\b")


def _row_values(row, label_re):
    """Cells of a table row: split on tabs or pipes, else take the text after the row label."""
    cells = [c.strip() for c in re.split(r"\t|\|", row) if c.strip()]
    if len(cells) >= 3:
        return cells[1], cells[2]
    m = label_re.search(row)
    if not m:
        return None
    rest = row[m.end():].strip()
    first = re.match(r"(no|none|not \w+|yes|required|unacknowledged|acknowledged)\b", rest)
    if not first:
        return None
    return first.group(0), rest[first.end():].strip()


def _table_ack(t):
    """Handle answers given as a table: a header row with trap/inform and an acknowledgment row."""
    lines = t.split("\n")
    for i, line in enumerate(lines):
        if "trap" in line and "inform" in line and line.index("trap") != line.index("inform"):
            trap_first = line.index("trap") < line.index("inform")
            label_re = re.compile(r"^\s*(acknowledg\w*|receiver response|confirmation|delivery confirmation|"
                                  r"reply|response)\b")
            for row in lines[i + 1:i + 15]:
                if not re.search(r"acknowledg|confirm|reliab|response", row):
                    continue
                vals = _row_values(row, label_re)
                if not vals:
                    continue
                a, b = vals
                trap_cell, inform_cell = (a, b) if trap_first else (b, a)
                no = re.compile(r"^(no|none|not|unacknowledged|unreliable)\b")
                yes = re.compile(r"^(yes|required|acknowledged|reliable|confirmed)|response pdu|acknowledg|confirm")
                if no.search(trap_cell) and yes.search(inform_cell):
                    return True, True
    return False, False


def check_trap_inform(answer, spec):
    t = normalize(answer).lower()
    trap_ok, inform_ok = _table_ack(t)
    trap_ok = trap_ok or any(re.search(p, t) for p in TRAP_NO_ACK)
    if not inform_ok:
        for s in sentences(t):
            for p in INFORM_ACK:
                m = re.search(p, s)
                if m and not re.search(r"not acknowledged|unacknowledged|no acknowledg|without", m.group(0)):
                    inform_ok = True
    errors = []
    for s in sentences(t):
        for p in TRAP_INFORM_WRONG:
            m = re.search(p, s)
            # "if an inform is not acknowledged, it is resent" is a condition, not a claim
            if m and not negated(s, m.start(), 15) and not CONDITION.search(s[max(0, m.start() - 20):m.start()]):
                errors.append("wrong claim: '%s'" % m.group(0))
    kps = [kp("Trap is not acknowledged", trap_ok),
           kp("Inform is acknowledged and resent until it is", inform_ok)]
    return kps, errors, []


# ------------------------------------------------ TC-QA-06 SNMPv3 levels

RECOMMEND = re.compile(r"recommend|should|best|production|go with|choose|pick|advis|always use")


def check_snmpv3_levels(answer, spec):
    t = normalize(answer).lower()
    squash = re.sub(r"[\s_\-]", "", t)
    n_noauth = squash.count("noauthnopriv")
    n_authno = squash.count("authnopriv") - n_noauth
    n_authpriv = squash.count("authpriv")
    protects = bool(re.search(r"authenticat|integrity|hmac|\bsha(-?\d+)?\b|md5", t)) and \
        bool(re.search(r"encrypt|privacy|confidential|\baes\b|\bdes\b", t))
    recommended, errors = False, []
    for s in sentences(t):
        sq = re.sub(r"[\s_\-]", "", s)
        if not RECOMMEND.search(s):
            continue
        has_ap = "authpriv" in sq
        has_weak = "noauthnopriv" in sq or (sq.count("authnopriv") - sq.count("noauthnopriv")) > 0
        if has_ap:
            recommended = True
        elif has_weak and re.search(r"recommend|should use|best choice|go with|production", s) \
                and not re.search(r"never|avoid|not\b|n't", s):
            errors.append("recommends a weaker level: '%s'" % s[:90])
    if recommended:
        errors = []
    kps = [kp("Names noAuthNoPriv", n_noauth > 0),
           kp("Names authNoPriv", n_authno > 0),
           kp("Names authPriv", n_authpriv > 0),
           kp("Explains what each protects (authentication, encryption)", protects),
           kp("Recommends authPriv for production", recommended)]
    return kps, errors, []


# ----------------------------------------------- TC-QA-07 net-snmp command

TOOLS = ("snmpwalk", "snmpbulkwalk", "snmptable")
NETSNMP_OPTIONS = set("aAcCdDeEhHIlLmMnOPrtTuvVxXYzZ3")
TABLE_TOKENS = {"if-mib::iftable", "iftable", "1.3.6.1.2.1.2.2", "iso.3.6.1.2.1.2.2",
                "if-mib::interfaces", "interfaces", "1.3.6.1.2.1.2", "mib-2.interfaces",
                "if-mib::iftable.ifentry", "ifentry", "1.3.6.1.2.1.2.2.1"}


def _commands(text):
    cmds = []
    for line in text.replace("`", "\n").split("\n"):
        for m in re.finditer(r"\b(%s)\b[^\n]*" % "|".join(TOOLS), line):
            cmds.append(m.group(0).strip())
    return cmds


def _tokens(cmd):
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    return [tok.rstrip(".,;:)") if not tok.endswith("::") else tok for tok in toks]


def _check_command(cmd, spec):
    toks = _tokens(cmd)
    tool = toks[0]
    want_v = spec.get("version", "2c")
    res = {"cmd": cmd, "tool": tool, "version": None, "community": None, "target": False,
           "table": False, "unknown_opts": [], "bulk_v1": False}
    i = 1
    while i < len(toks):
        tok = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if tok.startswith("--"):
            i += 1
            continue
        if tok.startswith("-") and len(tok) > 1 and not re.match(r"^-\d", tok):
            letter = tok[1]
            if letter not in NETSNMP_OPTIONS:
                res["unknown_opts"].append(tok)
            if letter == "v":
                res["version"] = tok[2:] if len(tok) > 2 else nxt
                i += 1 if len(tok) > 2 else 2
                continue
            if letter == "c":
                res["community"] = tok[2:] if len(tok) > 2 else nxt
                i += 1 if len(tok) > 2 else 2
                continue
            i += 1
            continue
        if re.fullmatch(r"(udp:)?%s(:161)?" % re.escape(spec["ip"]), tok):
            res["target"] = True
        if tok.lower().lstrip(".") in TABLE_TOKENS:
            res["table"] = True
        i += 1
    res["version_ok"] = (res["version"] or "").lower() == want_v
    res["community_ok"] = res["community"] == spec["community"]
    res["bulk_v1"] = tool == "snmpbulkwalk" and (res["version"] or "") == "1"
    res["score"] = sum([res["version_ok"], res["community_ok"], res["target"], res["table"]])
    return res


def check_snmp_command(answer, spec):
    t = normalize(answer)
    cands = [_check_command(c, spec) for c in _commands(t)]
    if not cands:
        kps = [kp("Gives a net-snmp walk/table command", False, "no command found"),
               kp("Right version flag (-v%s)" % spec.get("version", "2c"), False),
               kp("Right community (-c %s)" % spec["community"], False),
               kp("Right target %s" % spec["ip"], False),
               kp("Reads ifTable (IF-MIB::ifTable or 1.3.6.1.2.1.2.2)", False)]
        return kps, [], []
    best = max(cands, key=lambda c: (c["score"] - len(c["unknown_opts"]) - c["bulk_v1"]))
    errors, fabs = [], []
    if best["version"] is None:
        errors.append("no -v version flag, so the command does not run as SNMP%s" % spec.get("version"))
    elif not best["version_ok"]:
        errors.append("wrong version flag -v%s" % best["version"])
    if best["community"] is None:
        errors.append("no -c community string")
    elif not best["community_ok"]:
        errors.append("wrong community '%s'" % best["community"])
    if not best["target"]:
        errors.append("target %s missing" % spec["ip"])
    if best["bulk_v1"]:
        errors.append("snmpbulkwalk needs SNMPv2c, it does not work with -v1")
    if best["unknown_opts"]:
        fabs.append("invented option(s) %s" % " ".join(best["unknown_opts"]))
    kps = [kp("Gives a net-snmp walk/table command", True, best["cmd"][:120]),
           kp("Right version flag (-v%s)" % spec.get("version", "2c"), best["version_ok"]),
           kp("Right community (-c %s)" % spec["community"], best["community_ok"]),
           kp("Right target %s" % spec["ip"], best["target"]),
           kp("Reads ifTable (IF-MIB::ifTable or 1.3.6.1.2.1.2.2)", best["table"])]
    return kps, errors, fabs


# --------------------------------------------- TC-QA-08 nonexistent subject

def _subject_re(subject):
    m = re.match(r"snmpv(\d+)", subject.lower())
    if m:
        return r"snmp\s*-?\s*v\s*%s\b|snmp\s+version\s+%s\b" % (m.group(1), m.group(1))
    return re.escape(subject.lower())


def check_nonexistent(answer, spec):
    t = normalize(answer).lower()
    subj = _subject_re(spec["subject"])
    rejected = any(re.search(p, t) for p in [
        r"(there is|there's|there are|there has been|there was)\s+no\s+(official\s+|such\s+(thing\s+as\s+)?)?"
        r"(an?\s+)?(standardi[sz]ed\s+|official\s+)?(%s)" % subj,
        r"(%s)[^.\n]{0,60}?\b(does not|doesn't|did not|didn't|has not|hasn't|have not|haven't|was never|"
        r"were never|is not|isn't|never)\b[^.\n]{0,40}?(exist|been|defined|standardi|released|published|"
        r"real|official|ratified|developed|a thing|an? (official|real))" % subj,
        r"\bno\s+(rfc|standard|specification|official)[^.\n]{0,60}?(%s)" % subj,
        r"(%s)[^.\n]{0,30}?(non-?existent|not a real|fictional|hypothetical|doesn't exist|does not exist)" % subj,
        r"\b(it|this|that)\s+(does not|doesn't|did not|didn't)\s+exist",
    ])
    v3_latest = bool(re.search(r"snmp\s*-?\s*v\s*3|snmp\s+version\s+3", t)) and \
        bool(re.search(r"latest|newest|most recent|current|last|still the|remains|rfc\s*341[1-8]|std\s*62", t))
    fabs = []
    verb = r"(defin\w*|specif\w*|introduc\w*|describ\w*|publish\w*|standardi[sz]\w*)"
    # the RFC has to be tied to the subject itself: "SNMPv4 is defined in RFC 9561" or
    # "RFC 9561 defines SNMPv4" (an RFC that defines something else is not a fabrication)
    rfc_claims = [r"(%s)[^.\n]{0,40}?\b%s\s+(in|by)\s+rfc\s*\d{3,5}" % (subj, verb),
                  r"rfc\s*\d{3,5}[^.\n]{0,30}?\b%s\s+(the\s+)?(%s)" % (verb, subj)]
    for s in sentences(t):
        if not re.search(subj, s):
            continue
        claim = next((m for p in rfc_claims for m in [re.search(p, s)] if m), None)
        if claim and not NEGATION.search(s[:claim.end()]) and not re.search(r"rather than|instead of", s):
            fabs.append("claims an RFC defines %s: '%s'" % (spec["subject"], s[:100]))
        elif not rejected and not NEGATION.search(s) and \
                re.search(r"introduc|added|adds|new features?|brings|brought|improve", s):
            fabs.append("describes features of %s: '%s'" % (spec["subject"], s[:100]))
    kps = [kp("Says %s does not exist" % spec["subject"], rejected),
           kp("Points to SNMPv3 (RFC 3411-3418) as the latest version", v3_latest)]
    return kps, [], fabs


# ------------------------------------------------------------- dispatcher

ORACLES = {
    "getbulk_version": check_getbulk_version,
    "oid_unit": check_oid_unit,
    "counter_wrap": check_counter_wrap,
    "udp_reasons": check_udp_reasons,
    "trap_inform": check_trap_inform,
    "snmpv3_levels": check_snmpv3_levels,
    "snmp_command": check_snmp_command,
    "nonexistent": check_nonexistent,
}

# number of wrong-answer / fabrication rules per oracle (used for the complexity table)
ERROR_RULES = {"getbulk_version": 1, "oid_unit": 2, "counter_wrap": 2, "udp_reasons": 2, "trap_inform": 3,
               "snmpv3_levels": 1, "snmp_command": 7, "nonexistent": 2}


def evaluate(answer, test):
    """Grade one answer for one generated test case. Returns a dict that is stored in the results."""
    spec = test["oracle"]
    if not (answer or "").strip():
        return {"oracle": spec["type"], "output_class": "QA-OC-07", "verdict": "Fail",
                "key_points": [], "errors": ["empty answer"], "fabrications": [], "words": 0}
    kps, errors, fabs = ORACLES[spec["type"]](answer, spec)
    words = word_count(answer)
    oc = classify(kps, errors, fabs, words, test.get("max_words", 600),
                  nonexistent=test.get("expected_output") == "QA-OC-02")
    return {"oracle": spec["type"], "output_class": oc, "verdict": OC_VERDICT[oc],
            "key_points": kps, "errors": errors, "fabrications": fabs, "words": words}
