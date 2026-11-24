"""Lexical feature extraction for malicious-URL detection.

Everything here works on the URL *string only* - no DNS, WHOIS or page
download - so a URL can be scored in microseconds without ever contacting
the (possibly dangerous) site.

Public API
----------
normalize_url(url)        -> canonical string used for dedup + features
registered_domain(url)    -> "example.co.uk" / IP address (used to group URLs by site)
extract_features(url)     -> list[float] in the order of FEATURE_NAMES
extract_many(urls)        -> float32 matrix, computed in parallel
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tldextract

# Use the public-suffix list bundled with tldextract: no network access, and
# results are reproducible on any machine.
_TLD = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://")
_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
_INT_HOST_RE = re.compile(r"^(?:0x[0-9a-f]+|\d{8,10})$")  # hex / decimal encoded IP
_DIGIT_RUN_RE = re.compile(r"\d+")
_EXT_RE = re.compile(r"\.([a-zA-Z0-9]{1,5})$")
_HEX_ESC_RE = re.compile(r"%[0-9a-fA-F]{2}")

COMMON_TLDS = frozenset({"com", "org", "net", "edu", "gov", "mil", "int"})

# TLDs that abuse reports (Spamhaus, Interisle, APWG) repeatedly rank as
# disproportionately used for phishing/malware - mostly free or very cheap ones.
SUSPICIOUS_TLDS = frozenset({
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "club", "work", "click", "link",
    "loan", "win", "bid", "men", "date", "stream", "download", "racing", "review",
    "party", "science", "cricket", "accountant", "faith", "trade", "webcam", "icu",
    "buzz", "cam", "rest", "surf", "monster", "cyou", "cfd", "sbs", "quest", "zip",
    "mov", "country", "kim", "gdn", "pw", "cc", "ws", "info", "online", "site",
    "live", "support", "vip", "shop", "fit", "life",
})

SHORTENERS = frozenset({
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly",
    "bit.do", "cutt.ly", "rb.gy", "rebrand.ly", "shorturl.at", "tiny.cc", "t.ly",
    "v.gd", "x.co", "lnkd.in", "db.tt", "qr.ae", "j.mp", "tr.im", "cli.gs", "u.to",
    "s.id", "shorte.st", "bc.vc", "soo.gd", "clck.ru", "s2r.co", "urlz.fr",
})

# Words that social-engineering URLs use to look official or urgent.
SUSPICIOUS_WORDS = (
    "login", "log-in", "signin", "sign-in", "logon", "verify", "verification",
    "account", "update", "secure", "security", "confirm", "password", "passwd",
    "banking", "wallet", "webscr", "billing", "invoice", "recover", "unlock",
    "suspend", "authenticate", "credential", "wp-admin", "wp-includes",
    "wp-content", "cmd=", "admin", "bonus", "gift", "prize", "winner", "free",
    "support", "alert", "validate", "ebayisapi", "limited", "restore",
)
_SUSP_RE = re.compile("|".join(re.escape(w) for w in SUSPICIOUS_WORDS))

# Frequently impersonated brands. The feature only fires when the brand name
# appears somewhere *other than* the registered domain itself
# (paypal.com.evil.xyz -> 1, paypal.com/signin -> 0).
BRANDS = (
    "paypal", "apple", "icloud", "google", "gmail", "microsoft", "office365",
    "outlook", "amazon", "facebook", "instagram", "netflix", "whatsapp", "ebay",
    "chase", "wellsfargo", "bankofamerica", "dhl", "fedex", "usps", "linkedin",
    "dropbox", "adobe", "yahoo", "steam", "coinbase", "binance", "santander",
    "hsbc", "barclays", "alibaba", "dhl", "docusign", "onedrive", "sharepoint",
    "metamask", "blockchain", "telegram", "twitter", "spotify", "roblox",
)
_BRAND_RE = re.compile("|".join(sorted(set(BRANDS), key=len, reverse=True)))

EXEC_EXT = frozenset({"exe", "apk", "bin", "sh", "dll", "msi", "bat", "scr", "jar", "elf",
                      "ps1", "vbs", "dmg", "mips", "arm", "arm7", "x86", "mpsl", "i586",
                      "i686", "ppc", "sh4", "m68k", "spc", "com", "pif", "hta"})
SCRIPT_EXT = frozenset({"php", "asp", "aspx", "jsp", "cgi", "pl", "cfm", "do", "php3", "php5"})
HTML_EXT = frozenset({"html", "htm", "shtml", "xhtml"})
ARCHIVE_EXT = frozenset({"zip", "rar", "gz", "7z", "tar", "tgz", "iso", "img", "doc", "docm",
                         "xls", "xlsm"})

FEATURE_NAMES = [
    # --- whole URL ---
    "url_length", "n_dots", "n_hyphens", "n_underscores", "n_slashes", "n_question",
    "n_equals", "n_ampersand", "n_at", "n_percent", "n_digits", "n_letters",
    "n_special", "digit_ratio", "special_ratio", "uppercase_ratio", "url_entropy",
    "longest_digit_run", "n_hex_escapes", "has_embedded_url", "n_suspicious_words",
    "brand_outside_domain",
    # --- host ---
    "host_length", "host_is_ip", "has_port", "has_userinfo", "n_subdomains",
    "domain_length", "tld_length", "host_n_digits", "host_n_hyphens", "host_entropy",
    "host_longest_label", "domain_digit_ratio", "domain_vowel_ratio", "tld_is_common",
    "tld_is_country", "tld_is_suspicious", "is_shortener", "has_punycode",
    # --- path ---
    "path_length", "path_depth", "path_longest_segment", "path_n_digits",
    "path_has_double_slash", "ext_exec", "ext_script", "ext_html", "ext_archive",
    # --- query / fragment ---
    "query_length", "n_query_params", "query_longest_value", "has_fragment",
]
N_FEATURES = len(FEATURE_NAMES)


def normalize_url(url: str) -> str:
    """Canonical form: no scheme, no leading 'www.', lower-case host, no bare
    trailing slash. See TECHNICAL_DOCUMENTATION.md ("Normalisation") for why."""
    u = str(url).strip().strip("'\"")
    u = _SCHEME_RE.sub("", u, count=1)
    if u[:4].lower() == "www.":
        u = u[4:]
    # lower-case the authority part only (paths are case sensitive)
    cut = len(u)
    for ch in "/?#":
        i = u.find(ch)
        if i != -1 and i < cut:
            cut = i
    u = u[:cut].lower() + u[cut:]
    if u.endswith("/") and u.count("/") == 1:
        u = u[:-1]
    return u


def split_url(u: str) -> tuple[str, str, str, str, str, str]:
    """Split a *normalised* URL into (userinfo, host, port, path, query, fragment)."""
    rest, _, fragment = u.partition("#")
    rest, _, query = rest.partition("?")
    authority, slash, path = rest.partition("/")
    path = slash + path
    userinfo = ""
    if "@" in authority:
        userinfo, _, authority = authority.rpartition("@")
    host, port = authority, ""
    if not authority.startswith("["):  # not an IPv6 literal
        h, sep, p = authority.rpartition(":")
        if sep and p.isdigit():
            host, port = h, p
    return userinfo, host, port, path, query, fragment


def _is_ip(host: str) -> bool:
    return bool(_IPV4_RE.match(host) or _INT_HOST_RE.match(host) or host.startswith("["))


def registered_domain(url: str) -> str:
    """Site identifier: registrable domain (example.co.uk) or the IP/host itself."""
    host = split_url(normalize_url(url))[1]
    if not host or _is_ip(host):
        return host
    ext = _TLD.extract_str(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return host


def _entropy(s: str) -> float:
    n = len(s)
    if n < 2:
        return 0.0
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def extract_features(url: str) -> list[float]:
    u = normalize_url(url)
    userinfo, host, port, path, query, fragment = split_url(u)
    n = len(u) or 1

    n_digits = sum(c.isdigit() for c in u)
    n_letters = sum(c.isalpha() for c in u)
    n_upper = sum(c.isupper() for c in u)
    n_special = len(u) - n_digits - n_letters
    low = u.lower()

    # ---- host ----
    is_ip = _is_ip(host)
    if is_ip or not host:
        sub, dom, suffix = "", host, ""
    else:
        ext = _TLD.extract_str(host)
        sub, dom, suffix = ext.subdomain, ext.domain, ext.suffix
        if not suffix and "." in host:  # unknown TLD: fall back to last label
            parts = host.split(".")
            sub, dom, suffix = ".".join(parts[:-2]), parts[-2], parts[-1]
    last_label = suffix.rsplit(".", 1)[-1]
    labels = host.split(".") if host else [""]
    dom_len = len(dom) or 1
    reg_domain = f"{dom}.{suffix}" if suffix else dom

    # ---- path / query ----
    segments = [s for s in path.split("/") if s]
    last_seg = segments[-1] if segments else ""
    m = _EXT_RE.search(last_seg)
    ext_ = m.group(1).lower() if m else ""
    params = [p for p in query.split("&") if p] if query else []

    brand_hits = _BRAND_RE.findall(low)
    brand_outside = 0.0
    if brand_hits:
        # ignore the hit if it is the registered domain's own name
        outside = low.replace(reg_domain, "", 1) if dom in brand_hits else low
        brand_outside = float(bool(_BRAND_RE.search(outside)))

    after_host = u[len(host):]

    return [
        # whole URL
        float(len(u)),
        float(u.count(".")),
        float(u.count("-")),
        float(u.count("_")),
        float(u.count("/")),
        float(u.count("?")),
        float(u.count("=")),
        float(u.count("&")),
        float(u.count("@")),
        float(u.count("%")),
        float(n_digits),
        float(n_letters),
        float(n_special),
        n_digits / n,
        n_special / n,
        n_upper / n,
        _entropy(u),
        float(max((len(r) for r in _DIGIT_RUN_RE.findall(u)), default=0)),
        float(len(_HEX_ESC_RE.findall(u))),
        float("http" in after_host.lower() or "www." in after_host.lower()),
        float(len(set(_SUSP_RE.findall(low)))),
        brand_outside,
        # host
        float(len(host)),
        float(is_ip),
        float(bool(port)),
        float(bool(userinfo)),
        float(sub.count(".") + 1 if sub else 0),
        float(len(dom)),
        float(len(suffix)),
        float(sum(c.isdigit() for c in host)),
        float(host.count("-")),
        _entropy(host),
        float(max(len(x) for x in labels)),
        sum(c.isdigit() for c in dom) / dom_len,
        sum(c in "aeiou" for c in dom) / dom_len,
        float(last_label in COMMON_TLDS),
        float(len(last_label) == 2 and last_label.isalpha()),
        float(last_label in SUSPICIOUS_TLDS),
        float(reg_domain in SHORTENERS),
        float("xn--" in host),
        # path
        float(len(path)),
        float(len(segments)),
        float(max((len(s) for s in segments), default=0)),
        float(sum(c.isdigit() for c in path)),
        float("//" in path),
        float(ext_ in EXEC_EXT),
        float(ext_ in SCRIPT_EXT),
        float(ext_ in HTML_EXT),
        float(ext_ in ARCHIVE_EXT),
        # query / fragment
        float(len(query)),
        float(len(params)),
        float(max((len(p.partition("=")[2]) for p in params), default=0)),
        float(bool(fragment)),
    ]


def _extract_chunk(urls: list[str]) -> np.ndarray:
    return np.array([extract_features(u) for u in urls], dtype=np.float32).reshape(-1, N_FEATURES)


def _domain_chunk(urls: list[str]) -> list[str]:
    return [registered_domain(u) for u in urls]


def _parallel(fn, urls, n_jobs: int | None, chunk: int):
    urls = list(urls)
    chunks = [urls[i:i + chunk] for i in range(0, len(urls), chunk)]
    n_jobs = n_jobs or max(1, (os.cpu_count() or 2) - 2)
    if len(chunks) <= 1 or n_jobs == 1:
        return [fn(c) for c in chunks]
    with ProcessPoolExecutor(max_workers=min(n_jobs, len(chunks))) as pool:
        return list(pool.map(fn, chunks))


def extract_many(urls, n_jobs: int | None = None, chunk: int = 20_000) -> np.ndarray:
    """Feature matrix (n_urls, N_FEATURES), float32. Uses several processes for big inputs."""
    parts = _parallel(_extract_chunk, urls, n_jobs, chunk)
    return np.vstack(parts) if parts else np.empty((0, N_FEATURES), dtype=np.float32)


def registered_domains(urls, n_jobs: int | None = None, chunk: int = 50_000) -> list[str]:
    out: list[str] = []
    for part in _parallel(_domain_chunk, urls, n_jobs, chunk):
        out.extend(part)
    return out
