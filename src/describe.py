"""Plain-language text for the demo page: feature labels, modes and example URLs.

The page never hard-codes a feature or a mode. It shows whatever the saved model
contains and looks the wording up here. Anything missing from this file still
appears, with a label made from its name, so adding a feature or a threshold to
the model needs no change to the page - a description here is optional polish.
"""
from __future__ import annotations

from predict import MODES

# kind tells the page how to print a value:
#   count  whole number          chars  length in characters
#   flag   yes / no              share  0-1 shown as a percentage
#   bits   entropy in bits       score  learned score, shown with a sign
GROUPS = {
    "url": "Whole URL",
    "host": "Host",
    "path": "Path",
    "query": "Query and fragment",
    "learned": "Learned from training data",
    "other": "Other",
}

_F = {
    # ---- whole URL ----
    "url_length": ("url", "URL length", "chars", "Characters in the whole URL."),
    "n_dots": ("url", "Dots", "count", "How many '.' the URL contains."),
    "n_hyphens": ("url", "Hyphens", "count", "How many '-' the URL contains."),
    "n_underscores": ("url", "Underscores", "count", "How many '_' the URL contains."),
    "n_slashes": ("url", "Slashes", "count", "How many '/' the URL contains."),
    "n_question": ("url", "Question marks", "count", "How many '?' the URL contains."),
    "n_equals": ("url", "Equals signs", "count", "How many '=' the URL contains."),
    "n_ampersand": ("url", "Ampersands", "count", "How many '&' the URL contains."),
    "n_at": ("url", "@ signs", "count", "How many '@' the URL contains."),
    "n_percent": ("url", "Percent signs", "count", "How many '%' the URL contains."),
    "n_digits": ("url", "Digits", "count", "How many digits the URL contains."),
    "n_letters": ("url", "Letters", "count", "How many letters the URL contains."),
    "n_special": ("url", "Symbols", "count", "Characters that are neither letters nor digits."),
    "digit_ratio": ("url", "Share of digits", "share", "Digits as a share of all characters."),
    "special_ratio": ("url", "Share of symbols", "share", "Symbols as a share of all characters."),
    "uppercase_ratio": ("url", "Share of capitals", "share",
                        "Capital letters as a share of all characters."),
    "url_entropy": ("url", "Randomness of the URL", "bits",
                    "Shannon entropy of the characters. Random-looking strings score high."),
    "longest_digit_run": ("url", "Longest run of digits", "chars",
                          "Length of the longest unbroken string of digits."),
    "n_hex_escapes": ("url", "Encoded characters", "count",
                      "How many %xx escapes the URL contains."),
    "has_embedded_url": ("url", "Another address inside", "flag",
                         "'http' or 'www.' appears again after the host."),
    "n_suspicious_words": ("url", "Bait words", "count",
                           "Distinct words such as login, verify, account or free."),
    "brand_outside_domain": ("url", "Brand name outside the real domain", "flag",
                             "A well-known brand appears somewhere other than the registered "
                             "domain, as in paypal.com.example.xyz."),
    # ---- host ----
    "host_length": ("host", "Host length", "chars", "Characters in the host name."),
    "host_is_ip": ("host", "Host is an IP address", "flag",
                   "The host is a number such as 194.48.152.10 instead of a name."),
    "has_port": ("host", "Port number given", "flag", "The host is followed by ':' and a port."),
    "has_userinfo": ("host", "User name before the host", "flag",
                     "Text before an '@' in the host, which can disguise the real site."),
    "n_subdomains": ("host", "Subdomain levels", "count",
                     "How many levels sit in front of the registered domain."),
    "domain_length": ("host", "Domain name length", "chars",
                      "Characters in the registered name, without its ending."),
    "tld_length": ("host", "Ending length", "chars",
                   "Characters in the ending, such as com or co.uk."),
    "host_n_digits": ("host", "Digits in the host", "count", "How many digits the host contains."),
    "host_n_hyphens": ("host", "Hyphens in the host", "count", "How many '-' the host contains."),
    "host_entropy": ("host", "Randomness of the host", "bits",
                     "Shannon entropy of the host's characters."),
    "host_longest_label": ("host", "Longest piece of the host", "chars",
                           "Length of the longest part between dots."),
    "domain_digit_ratio": ("host", "Share of digits in the domain", "share",
                           "Digits as a share of the registered name."),
    "domain_vowel_ratio": ("host", "Share of vowels in the domain", "share",
                           "Vowels as a share of the registered name. Made-up names often have few."),
    "tld_is_common": ("host", "Long-established ending", "flag",
                      "The ending is com, org, net, edu, gov, mil or int."),
    "tld_is_country": ("host", "Country ending", "flag", "The ending is a two-letter country code."),
    "tld_is_suspicious": ("host", "Ending often abused", "flag",
                          "The ending is one that abuse reports repeatedly rank as heavily misused."),
    "is_shortener": ("host", "Link shortener", "flag",
                     "The site is a known shortener such as bit.ly."),
    "has_punycode": ("host", "Encoded international name", "flag",
                     "The host contains 'xn--', used for look-alike characters."),
    # ---- path ----
    "path_length": ("path", "Path length", "chars", "Characters in the path."),
    "path_depth": ("path", "Folders deep", "count", "How many segments the path has."),
    "path_longest_segment": ("path", "Longest path segment", "chars",
                             "Length of the longest part between slashes."),
    "path_n_digits": ("path", "Digits in the path", "count", "How many digits the path contains."),
    "path_has_double_slash": ("path", "Double slash in the path", "flag",
                              "The path contains '//'."),
    "ext_exec": ("path", "Ends in a program file", "flag",
                 "The last segment ends in an extension such as exe, apk or sh."),
    "ext_script": ("path", "Ends in a server script", "flag",
                   "The last segment ends in an extension such as php or asp."),
    "ext_html": ("path", "Ends in a web page file", "flag",
                 "The last segment ends in html or htm."),
    "ext_archive": ("path", "Ends in an archive or document", "flag",
                    "The last segment ends in an extension such as zip, rar or doc."),
    # ---- query / fragment ----
    "query_length": ("query", "Query length", "chars", "Characters after the '?'."),
    "n_query_params": ("query", "Query parameters", "count", "How many parameters the query has."),
    "query_longest_value": ("query", "Longest query value", "chars",
                            "Length of the longest parameter value."),
    "has_fragment": ("query", "Has a fragment", "flag", "The URL has a '#' part."),
    # ---- learned ----
    "tld_malicious_rate": ("learned", "Track record of this ending", "share",
                           "Share of training sites with this ending that were malicious."),
    "host_ngram_score": ("learned", "Host character patterns", "score",
                         "How much the host's short letter sequences resemble malicious hosts "
                         "seen in training, rather than benign ones."),
    "url_ngram_score": ("learned", "URL character patterns", "score",
                        "How much the URL's short letter sequences resemble malicious URLs "
                        "seen in training, rather than benign ones."),
}

_MODE_TEXT = {
    "balanced": ("Balanced", "The cut-off that gave the best balance between catching "
                             "malicious URLs and raising false alarms on validation sites."),
    "strict": ("Few false alarms", "Flags a URL only when the model is very sure, so few "
                                   "safe URLs are flagged and more malicious ones slip through."),
    "default": ("Plain 0.5", "Flags a URL when more than half the weighted vote says malicious."),
}

# Shown as one-click examples. The model reads the text only; nothing is ever fetched.
EXAMPLES = [
    {"label": "Lookalike sign-in", "url": "http://paypal.com.secure-login-verify.xyz/webscr?cmd=_login"},
    {"label": "Malware download", "url": "http://194.48.152.10/bins/x86.sh"},
    {"label": "Prize scam", "url": "http://free-gift-card-winner.tk/claim.php"},
    {"label": "Ordinary page", "url": "github.com/scikit-learn/scikit-learn"},
    {"label": "A real sign-in page it gets wrong", "url": "https://accounts.google.com/signin/v2/identifier"},
]


# A made-up phishing email for the inbox triage example. It mixes the tricks the
# triage step handles: an HTML link whose text names another site, a defanged link,
# a lookalike domain, a bare address and harmless links. Nothing in it is fetched.
EXAMPLE_MESSAGE = """From: "PayPal Service" <service@paypa1-support.com>
Subject: Action required: your account has been limited

Dear customer,

We noticed unusual sign-in activity and have temporarily limited your account.
To restore full access, confirm your details within 24 hours:

<a href="http://paypal.com.account-verify.secure-login.xyz/webscr?cmd=_login">https://www.paypal.com/myaccount</a>

If the button does not work, copy this address into your browser:
paypa1-support.com/restore/account?id=88217

Our security team has also flagged this file, please do not open it:
hxxp://194.48.152[.]10/bins/invoice.exe

Track your recent order at https://www.amazon.com/gp/css/order-history
Questions? Visit https://www.paypal.com/us/smarthelp/home or github.com/paypal

PayPal (c) 2026
"""


def _readable(name: str) -> str:
    return name.replace("_", " ").strip().capitalize()


def feature(name: str) -> dict:
    group, label, kind, text = _F.get(name, ("other", _readable(name), "number", ""))
    return {"name": name, "group": group, "label": label, "kind": kind, "description": text}


def modes(thresholds: dict) -> list[dict]:
    """One entry per decision threshold stored with the model."""
    out, used = [], set()
    for mode_id, key in MODES.items():
        if key in thresholds:
            label, text = _MODE_TEXT.get(mode_id, (_readable(mode_id), ""))
            out.append({"id": mode_id, "key": key, "threshold": float(thresholds[key]),
                        "label": label, "description": text})
            used.add(key)
    for key, value in thresholds.items():
        is_threshold = isinstance(value, (int, float)) and 0 <= value <= 1
        if key not in used and is_threshold and not key.endswith("_target"):
            out.append({"id": key, "key": key, "threshold": float(value),
                        "label": _readable(key), "description": ""})
    return out
