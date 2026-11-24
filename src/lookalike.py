"""Lookalike-domain check: does this address pretend to be a well-known brand?

    python src/lookalike.py paypa1.com "xn--pypal-4ve.com/signin" paypal.com.secure.xyz

The random forest learns patterns from 1.1 M training URLs, but typosquats such
as `paypa1.com` are rare in that data and look perfectly ordinary character by
character, so the forest scores them as safe. This module is a second, rule-based
layer that reads the same URL text (no network lookups) and looks for the tricks
attackers use to borrow a brand's trust:

    homoglyph      characters from another alphabet that look Latin (Cyrillic 'а')
                   or a punycode name (xn--...) that decodes to such characters
    character swap digits or letter pairs that read like a letter (0->o, 1->l, rn->m)
    typo           one or two letters missing, added, changed or swapped (Damerau-
                   Levenshtein distance), for brand names long enough to make that rare
    combo          the brand plus extra words (paypal-secure-login.com)
    subdomain      the brand's domain in front of someone else's (paypal.com.evil.xyz)
    userinfo       the brand before an '@', which the browser ignores
    ending         the brand's name under a different ending (paypal.co)

Its findings are shown next to the model's score, never mixed into it: the score
stays exactly the forest's, and the two layers can be judged separately.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import features as F

# brand key (as it appears in a domain) -> (display name, official registered domains)
BRANDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "paypal": ("PayPal", ("paypal.com", "paypal.me", "paypalobjects.com")),
    "apple": ("Apple", ("apple.com", "icloud.com", "apple.news")),
    "icloud": ("iCloud", ("icloud.com", "apple.com")),
    "google": ("Google", ("google.com", "goo.gl", "youtube.com", "gmail.com", "googleusercontent.com",
                          "googleapis.com", "gstatic.com", "blogger.com", "blogspot.com")),
    "gmail": ("Gmail", ("gmail.com", "google.com")),
    "microsoft": ("Microsoft", ("microsoft.com", "live.com", "office.com", "microsoftonline.com",
                                "outlook.com", "windows.net", "azure.com", "msn.com", "bing.com")),
    "office365": ("Office 365", ("office.com", "office365.com", "microsoft.com")),
    "outlook": ("Outlook", ("outlook.com", "live.com", "office.com", "microsoft.com")),
    "onedrive": ("OneDrive", ("onedrive.com", "live.com", "microsoft.com")),
    "sharepoint": ("SharePoint", ("sharepoint.com", "microsoft.com")),
    "amazon": ("Amazon", ("amazon.com", "amazon.in", "amazon.co.uk", "amazon.de", "amazon.co.jp",
                          "amazonaws.com", "amazon.ca", "amazon.fr", "amazon.it", "amazon.es")),
    "facebook": ("Facebook", ("facebook.com", "fb.com", "fbcdn.net", "messenger.com")),
    "instagram": ("Instagram", ("instagram.com",)),
    "whatsapp": ("WhatsApp", ("whatsapp.com", "whatsapp.net", "wa.me")),
    "netflix": ("Netflix", ("netflix.com",)),
    "ebay": ("eBay", ("ebay.com", "ebay.co.uk", "ebay.de", "ebay.in")),
    "chase": ("Chase", ("chase.com",)),
    "wellsfargo": ("Wells Fargo", ("wellsfargo.com",)),
    "bankofamerica": ("Bank of America", ("bankofamerica.com", "bofa.com")),
    "dhl": ("DHL", ("dhl.com", "dhl.de")),
    "fedex": ("FedEx", ("fedex.com",)),
    "usps": ("USPS", ("usps.com",)),
    "linkedin": ("LinkedIn", ("linkedin.com", "lnkd.in")),
    "dropbox": ("Dropbox", ("dropbox.com", "db.tt", "dropboxusercontent.com")),
    "adobe": ("Adobe", ("adobe.com",)),
    "yahoo": ("Yahoo", ("yahoo.com", "yahoo.co.jp")),
    "steam": ("Steam", ("steampowered.com", "steamcommunity.com")),
    "coinbase": ("Coinbase", ("coinbase.com",)),
    "binance": ("Binance", ("binance.com",)),
    "santander": ("Santander", ("santander.com", "santander.co.uk")),
    "hsbc": ("HSBC", ("hsbc.com", "hsbc.co.uk", "hsbc.co.in")),
    "barclays": ("Barclays", ("barclays.com", "barclays.co.uk")),
    "alibaba": ("Alibaba", ("alibaba.com", "aliexpress.com")),
    "docusign": ("DocuSign", ("docusign.com", "docusign.net")),
    "metamask": ("MetaMask", ("metamask.io",)),
    "blockchain": ("Blockchain.com", ("blockchain.com",)),
    "telegram": ("Telegram", ("telegram.org", "t.me")),
    "twitter": ("Twitter / X", ("twitter.com", "x.com", "t.co")),
    "spotify": ("Spotify", ("spotify.com",)),
    "roblox": ("Roblox", ("roblox.com",)),
    "github": ("GitHub", ("github.com", "github.io", "githubusercontent.com")),
    "sbi": ("State Bank of India", ("sbi.co.in", "onlinesbi.sbi", "onlinesbi.com")),
    "hdfcbank": ("HDFC Bank", ("hdfcbank.com",)),
    "icicibank": ("ICICI Bank", ("icicibank.com",)),
    "paytm": ("Paytm", ("paytm.com",)),
    "flipkart": ("Flipkart", ("flipkart.com",)),
}
OFFICIAL = {domain for _, domains in BRANDS.values() for domain in domains}

# Brand names that are also ordinary words or common fragments ("purchase", "apples").
# For these only an exact, separated match counts - never a prefix or a typo.
WORDLIKE = frozenset({"apple", "chase", "steam", "adobe", "outlook", "office365", "sbi", "dhl",
                      "usps", "hsbc", "ebay", "yahoo"})
MIN_TYPO_LENGTH = 6          # shorter brands are one typo away from too many real words

# Characters that are drawn (almost) like a Latin letter. Covers the Cyrillic and
# Greek letters seen in real IDN homograph attacks, a few Latin variants, and
# digits / symbols used in plain-ASCII swaps.
_CONFUSABLE = {
    # Cyrillic
    "а": "a", "в": "b", "с": "c", "ԁ": "d", "е": "e", "ё": "e", "һ": "h", "і": "i", "ї": "i",
    "ј": "j", "к": "k", "ӏ": "l", "м": "m", "п": "n", "о": "o", "р": "p", "ԛ": "q", "г": "r",
    "ѕ": "s", "т": "t", "ц": "u", "у": "y", "х": "x", "ү": "y", "ԝ": "w", "ь": "b",
    # Greek
    "α": "a", "β": "b", "ϲ": "c", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w",
    # Latin look-alikes
    "ı": "i", "ɡ": "g", "ł": "l", "ɑ": "a", "ǀ": "l", "ʟ": "l",
}
_DIGIT_SWAPS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "9": "g", "@": "a", "$": "s"}
_PAIR_SWAPS = (("rn", "m"), ("vv", "w"), ("cl", "d"), ("nn", "m"))

TECHNIQUES = {
    "homoglyph": ("Letters from another alphabet", "high",
                  "Some characters only look Latin. In the browser bar the name reads like {real}."),
    "swap": ("Look-alike characters", "high",
             "Digits or letter pairs stand in for letters, so the name reads like {real} at a glance."),
    "typo": ("One slip from the real name", "high",
             "The name is {n} keystroke{s} away from {real}, the kind of slip a reader does not notice."),
    "subdomain": ("Brand placed in front of another site", "high",
                  "The address starts with {real}, but the site it really belongs to is {site}."),
    "userinfo": ("Brand placed before an @", "high",
                 "Browsers ignore everything before the @, so this opens {site}, not {real}."),
    "combo": ("Brand name plus extra words", "medium",
              "The name contains {brand} with extra words around it. {brand} does not use this domain."),
    "path": ("Brand name only in the path", "low",
             "{brand} appears after the site name, which anyone can put in a path on any site."),
    "ending": ("Brand name under another ending", "low",
               "The name is {brand}'s, but the ending differs from {real}. Large brands own many "
               "endings, so this can be genuine."),
}


# --------------------------------------------------------------------- helpers

def _decode_host(host: str) -> str:
    """Show punycode labels (xn--...) as the Unicode a browser would display."""
    out = []
    for label in host.split("."):
        if label.startswith("xn--"):
            try:
                label = label.encode("ascii").decode("idna")
            except (UnicodeError, ValueError):
                pass
        out.append(label)
    return ".".join(out)


def _scripts(text: str) -> list[str]:
    found = []
    for ch in text:
        if ch.isalpha():
            script = unicodedata.name(ch, "UNKNOWN").split(" ")[0].capitalize()
            if script not in found:
                found.append(script)
    return found


def skeleton(text: str, digits: bool = True) -> str:
    """Map every look-alike character to the Latin letter it imitates."""
    text = unicodedata.normalize("NFKC", text.lower())
    out = "".join(_CONFUSABLE.get(c, _DIGIT_SWAPS.get(c, c) if digits else c) for c in text)
    for pair, single in _PAIR_SWAPS:
        out = out.replace(pair, single)
    return out


def edit_ops(seen: str, real: str) -> list[list[str]]:
    """Optimal-string-alignment (Damerau-Levenshtein) alignment of `seen` onto `real`.

    Returns [op, seen_char, real_char] with op in same / change / extra / missing / swap,
    which the page draws character by character.
    """
    n, m = len(seen), len(real)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = seen[i - 1] != real[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and seen[i - 1] == real[j - 2] and seen[i - 2] == real[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if (i > 1 and j > 1 and seen[i - 1] == real[j - 2] and seen[i - 2] == real[j - 1]
                and seen[i - 1] != seen[i - 2] and d[i][j] == d[i - 2][j - 2] + 1):
            ops.append(["swap", seen[i - 2:i], real[j - 2:j]])
            i, j = i - 2, j - 2
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (seen[i - 1] != real[j - 1]):
            ops.append(["same" if seen[i - 1] == real[j - 1] else "change", seen[i - 1], real[j - 1]])
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(["extra", seen[i - 1], ""])
            i -= 1
        else:
            ops.append(["missing", "", real[j - 1]])
            j -= 1
    return ops[::-1]


def distance(seen: str, real: str) -> int:
    return sum(op[0] != "same" for op in edit_ops(seen, real))


def _tokens(name: str) -> list[str]:
    out, word = [], ""
    for c in name:
        if c.isalpha():
            word += c
        else:
            if word:
                out.append(word)
            word = ""
    return out + ([word] if word else [])


def _finding(technique: str, brand: str, site: str, seen: str | None = None, **extra) -> dict:
    name, official = BRANDS[brand]
    label, severity, text = TECHNIQUES[technique]
    real = official[0]
    n = extra.pop("n", 0)
    found = {
        "technique": technique, "label": label, "severity": severity,
        "brand": name, "official": real,
        "detail": text.format(real=real, brand=name, site=site, n=n, s="" if n == 1 else "s"),
    }
    if seen is not None:
        found["compare"] = {"seen": seen, "real": brand, "ops": edit_ops(seen, brand)}
    found.update(extra)
    return found


# ------------------------------------------------------------------------- API

def check(url: str) -> dict:
    """Look for brand impersonation in one URL. Never contacts the site."""
    norm = F.normalize_url(url)
    userinfo, host, _port, path, query, _fragment = F.split_url(norm)
    display_host = _decode_host(host)
    result = {"host": host, "display_host": display_host,
              "scripts": _scripts(display_host), "findings": []}
    if not host or F._is_ip(host):
        return result

    site = F.registered_domain(norm)
    ext = F._TLD.extract_str(display_host)
    name, suffix, sub = ext.domain, ext.suffix, ext.subdomain
    if not name:
        return result
    shown_site = f"{name}.{suffix}" if suffix else name
    findings: list[dict] = []
    seen_brands: set[str] = set()

    def add(technique, brand, **kw):
        if brand not in seen_brands:
            seen_brands.add(brand)
            findings.append(_finding(technique, brand, shown_site, **kw))

    if site in OFFICIAL:
        # A real brand site. Only an @ trick can still mislead here.
        return result

    non_ascii = any(ord(c) > 127 for c in name)
    mixed = len(result["scripts"]) > 1
    # the whole name, and each hyphen-separated word of it (paypa1-support)
    words = [name] + ([w for w in name.split("-") if w] if "-" in name else [])
    mapped = [(w, skeleton(w, digits=False), skeleton(w)) for w in words]
    for brand in BRANDS:
        # 1. the name *is* the brand once look-alike characters are mapped back
        for word, plain, full in mapped:
            if word == brand:
                continue
            if non_ascii and (plain == brand or full == brand):
                add("homoglyph", brand, seen=word, mixed_scripts=mixed)
            elif full == brand and len(brand) >= 4:
                add("swap", brand, seen=word)

        # 2. exactly the brand's name under an ending it does not use
        if name == brand and suffix:
            add("ending", brand)

    joined = skeleton(name, digits=False)
    tokens = _tokens(joined)
    pieces = words if name.isascii() else []
    for brand in BRANDS:
        if brand in seen_brands:
            continue
        # 3. typo of a long brand name
        #    (the whole name, or one hyphen-separated word of it: micorsoft-support)
        if len(brand) >= MIN_TYPO_LENGTH and brand not in WORDLIKE:
            limit = 1 if len(brand) < 9 else 2
            hit = None
            for piece in pieces:
                # cheap filters first: lengths close, and most letters shared
                if abs(len(piece) - len(brand)) > limit or piece == brand:
                    continue
                if len(set(piece) ^ set(brand)) > 2 * limit:
                    continue
                n = distance(piece, brand)
                if 0 < n <= limit:
                    hit = (piece, n)
                    break
            if hit:
                add("typo", brand, seen=hit[0], n=hit[1])
                continue
        # 4. brand plus extra words
        if brand != name and len(brand) >= 4 and (
                brand in tokens
                or (brand not in WORDLIKE and len(brand) >= 5 and brand in joined)):
            add("combo", brand)

    # 5. brand or brand domain in the subdomain / before an @
    for where, text in (("userinfo", userinfo), ("subdomain", sub)):
        if not text:
            continue
        low = skeleton(text, digits=False)
        for brand in BRANDS:
            if brand in seen_brands:
                continue
            pieces = _tokens(low)
            if brand in pieces or (brand not in WORDLIKE and len(brand) >= 5 and brand in low):
                add(where, brand)

    # 6. brand only in the path or query
    rest = skeleton(path + "?" + query, digits=False)
    for brand in BRANDS:
        if brand not in seen_brands and brand not in WORDLIKE and len(brand) >= 5 and brand in rest:
            add("path", brand)

    order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda f: order[f["severity"]])
    result["findings"] = findings
    return result


def main() -> int:
    for url in sys.argv[1:] or ["paypa1.com"]:
        r = check(url)
        print(url, "->", r["display_host"], r["scripts"])
        for f in r["findings"]:
            print(f"   [{f['severity']}] {f['label']}: {f['detail']}")
        if not r["findings"]:
            print("   no impersonation found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
