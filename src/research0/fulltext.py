"""Full-text acquisition: open-access PDFs found by DOI (Unpaywall, then OpenAlex, then Semantic Scholar) plus PDFs uploaded by hand, kept in S3.

The database stores URI, checksum, origin, version and who acquired each file; the bytes live under fulltext/ in S3_BUCKET.
Only locations that Unpaywall, OpenAlex or Semantic Scholar declare open access are fetched, including their full-text links in HTML.
OpenAlex is asked only when Unpaywall gave no file, and it can point to repository copies (accepted manuscripts, preprints):
the version is stored with the file. Semantic Scholar (its openAccessPdf) is asked last, only with SEMANTIC_SCHOLAR_API_KEY.
"""
import csv
import hashlib
import http.client
import io
import ipaddress
import json
import logging
import math
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from pathlib import Path

LOG = logging.getLogger("research0")
UNPAYWALL = "https://api.unpaywall.org/v2/"
# OPENALEX_API_KEY is optional for basic use; a free key raises the API budget.
OPENALEX = "https://api.openalex.org/works/doi:"
OPENALEX_FIELDS = "id,open_access,best_oa_location,locations"
# SEMANTIC_SCHOLAR_API_KEY turns the third source on. The key allows 1 request per second across all endpoints, so every
# request of this process waits its turn (one at a time, SEMANTIC_SCHOLAR_INTERVAL apart, retries included).
SEMANTIC_SCHOLAR = "https://api.semanticscholar.org/graph/v1/paper/DOI:"
SEMANTIC_SCHOLAR_FIELDS = "isOpenAccess,openAccessPdf"
SEMANTIC_SCHOLAR_INTERVAL = 1.1
_semantic_scholar_lock = threading.Lock()
_semantic_scholar_next = 0.0
VERSIONS = {"publishedVersion": "published", "acceptedVersion": "accepted", "submittedVersion": "submitted"}
MAX_PDF_BYTES = 60 * 1024 * 1024
MAX_LOOKUP_BYTES = 2 * 1024 * 1024
WORKERS = 4
MAX_CANDIDATES = 12
MAX_PAGE_BYTES = 4 * 1024 * 1024
MAX_PAGE_DEPTH = 2
MAX_DOCUMENT_REQUESTS = 40
RESOLVER_VERSION = "oa-pages-v3"
ABORT_AFTER_LOOKUP_FAILURES = 20
SCOPES = ("sample", "audit", "candidates", "assigned", "eligible")
# Outcomes that a new run tries again; every other outcome is a final answer about that paper.
RETRY = ("lookup_failed", "download_failed")
# Recheck final answers from older resolvers once; transient failures always remain retryable.
RECHECK = ("not_open_access", "no_pdf_link", "not_a_pdf", "too_large")
REASONS = {
    "no_doi": "No DOI in the record",
    "not_open_access": "Not open access",
    "no_pdf_link": "Open access page, but no direct PDF link",
    "download_failed": "The publisher refused or failed the download",
    "blocked_by_site": "The site only serves people: bot protection (e.g. Cloudflare). Open the link in your browser and upload the PDF",
    "not_a_pdf": "The link did not return a PDF",
    "too_large": "PDF larger than the allowed size",
}
# `challenge` is true when the answer is a bot-protection wall (a JavaScript or CAPTCHA test), not a refusal about this paper.
Response = namedtuple("Response", "status url content_type body retry_after challenge", defaults=[None, False])
FIGSHARE = "https://api.figshare.com/v2/articles/"
CHALLENGE_STATUSES = (202, 401, 403, 405, 429, 503)
CHALLENGE_MARKERS = (b"just a moment...", b"human verification", b"site protection", b"cf-challenge", b"attention required! | cloudflare",
                     b"checking your browser", b"are you a robot", b"captcha")


class FetchError(Exception):
    pass


class TooLarge(FetchError):
    pass


# --- Network: public addresses only, checked at every redirect hop and again at connection time ---------------------

def _public_socket(host, port, timeout):
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    for *_, addr in infos:
        ip = ipaddress.ip_address(addr[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise urllib.error.URLError(f"refusing non-public address {ip} for {host}")
    error = OSError("no address")
    for family, kind, proto, _, addr in infos:
        sock = socket.socket(family, kind, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(addr)
            return sock
        except OSError as exc:
            sock.close()
            error = exc
    raise error


class _PublicHTTP(http.client.HTTPConnection):
    def connect(self):
        self.sock = _public_socket(self.host, self.port, self.timeout)


class _PublicHTTPS(http.client.HTTPSConnection):
    def connect(self):
        sock = _public_socket(self.host, self.port, self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PublicHTTP, req)


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_PublicHTTPS, req, context=self._context)


def _opener():
    # No proxy, file, ftp or data handlers: a redirect can only lead to another public http(s) address.
    opener = urllib.request.OpenerDirector()
    for handler in (_HTTPHandler(), _HTTPSHandler(), urllib.request.HTTPRedirectHandler(), urllib.request.HTTPDefaultErrorHandler(),
                    urllib.request.HTTPErrorProcessor(), urllib.request.UnknownHandler()):
        opener.add_handler(handler)
    return opener


def is_challenge(status, headers, body):
    """A bot-protection wall: named in a header, or a refusal page that carries a known challenge text. Plain refusals are not walls."""
    if (headers.get("cf-mitigated") or "").lower() == "challenge" or (headers.get("x-amzn-waf-action") or "").lower() in ("challenge", "captcha"):
        return True
    return status in CHALLENGE_STATUSES and any(marker in body[:50000].lower() for marker in CHALLENGE_MARKERS)


def _http_get_once(url, *, max_bytes, user_agent, accept="*/*", timeout=20, headers=None):
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": accept})
    for name, value in (headers or {}).items():
        request.add_unredirected_header(name, value)  # credentials never follow a redirect to another host
    try:
        try:
            response = _opener().open(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise TooLarge(f"more than {max_bytes} bytes")
            return Response(response.status, response.geturl(), response.headers.get("Content-Type", ""), body,
                            response.headers.get("Retry-After"), is_challenge(response.status, response.headers, body))
    except FetchError:
        raise
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as exc:
        raise FetchError(str(getattr(exc, "reason", exc))[:200]) from exc


def http_get(url, *, max_bytes, user_agent, accept="*/*", timeout=20, headers=None):
    """Retry transient errors only, with bounded backoff; never retry a refused/oversized download."""
    for attempt in range(3):
        try:
            reply = _http_get_once(url, max_bytes=max_bytes, user_agent=user_agent, accept=accept, timeout=timeout, headers=headers)
        except TooLarge:
            raise
        except FetchError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
            continue
        # Asking a bot-protection wall again only repeats the wall.
        if reply.challenge or reply.status not in (408, 429, 500, 502, 503, 504) or attempt == 2:
            return reply
        delay = 2 ** attempt
        if reply.retry_after:
            try:
                delay = max(delay, float(reply.retry_after))
            except ValueError:
                try:
                    delay = max(delay, (parsedate_to_datetime(reply.retry_after) - datetime.now(timezone.utc)).total_seconds())
                except (ValueError, TypeError, OverflowError):
                    pass
        # A long cooldown should be resumed in a later run, not ignored by retrying early.
        if delay > 30:
            return reply
        time.sleep(delay)


# --- Finding and validating the PDF -----------------------------------------------------------------------------------

def check_pdf(data):
    if not data:
        raise ValueError("The file is empty")
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"The file is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB")
    if b"%PDF-" not in data[:1024]:
        raise ValueError("The file is not a PDF")


def check_document(data):
    """Format of a file chosen by a person: 'pdf', 'docx' or 'doc', recognised from its content, never from its name."""
    if not data:
        raise ValueError("The file is empty")
    if len(data) > MAX_PDF_BYTES:
        raise ValueError(f"The file is larger than {MAX_PDF_BYTES // (1024 * 1024)} MB")
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":  # OLE2 container: the legacy Word format
        return "doc"
    if data[:4] == b"PK\x03\x04":
        import zipfile
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                if "word/document.xml" in archive.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
    raise ValueError("The file is not a PDF or a Word document")


def pdf_candidates(record):
    """Direct PDF links from an Unpaywall record, best location first."""
    urls = []
    for location in [record.get("best_oa_location"), *(record.get("oa_locations") or [])]:
        url = (location or {}).get("url_for_pdf")
        if url and url not in urls:
            urls.append(url)
    return urls[:MAX_CANDIDATES]


def openalex_candidates(record, skip=()):
    """Direct PDF links from an OpenAlex work as (url, version, license), best location first; links already tried are left out."""
    links = []
    for location in [record.get("best_oa_location"), *(record.get("locations") or [])]:
        location = location or {}
        url = location.get("pdf_url")
        if location.get("is_oa") and url and url not in skip and all(url != known[0] for known in links):
            links.append((url, VERSIONS.get(location.get("version")), location.get("license")))
    return links[:MAX_CANDIDATES]


def _web_url(value, base=""):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        url = urllib.parse.urljoin(base, value.strip())
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            return None
        return urllib.parse.urlunsplit(parts._replace(fragment=""))
    except ValueError:
        return None


def _doi(value):
    value = urllib.parse.unquote(str(value or "").strip())
    return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I).strip().lower()


class _PaperPage(HTMLParser):
    """Extract article metadata and explicit document links, never crawl a site's navigation."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.metadata, self.anchors, self.dois = [], [], []
        self.anchor = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        name = (attrs.get("name") or attrs.get("property") or "").lower()
        content = attrs.get("content") or ""
        if tag == "meta":
            if name in ("citation_doi", "dc.identifier", "dc.identifier.doi", "prism.doi"):
                if _doi(content).startswith("10."):
                    self.dois.append(_doi(content))
            if name in ("citation_pdf_url", "wkhealth_pdf_url", "eprints.document_url"):
                self.metadata.append(content)
        elif tag == "link" and (attrs.get("type") or "").lower() == "application/pdf":
            if "alternate" in (attrs.get("rel") or "").lower().split():
                self.metadata.append(attrs.get("href"))
        elif tag in ("iframe", "embed", "object"):
            url = attrs.get("src") or attrs.get("data") or ""
            if ".pdf" in url.lower() or attrs.get("type") == "application/pdf":
                # PDF.js viewers expose the actual document in their file query parameter.
                try:
                    url = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("file", [url])[0]
                except ValueError:
                    return
                self.metadata.append(url)
        elif tag == "a":
            self.anchor = [attrs.get("href") or "", attrs.get("title") or ""]

    def handle_data(self, data):
        if self.anchor is not None:
            self.anchor[1] += " " + data

    def handle_endtag(self, tag):
        if tag == "a" and self.anchor is not None:
            self.anchors.append(self.anchor)
            self.anchor = None


def page_pdf_links(body, page_url, doi=None):
    """Conservative, bounded extraction from a known OA page; reject a page for a different DOI."""
    if len(body) > MAX_PAGE_BYTES:
        return []
    parser = _PaperPage()
    parser.feed(body.decode("utf-8", errors="replace"))
    if doi and parser.dois and _doi(doi) not in parser.dois:
        return []
    excluded = re.compile(r"supplement|supporting|appendi[xc]|cover.?sheet|terms.of.use", re.I)
    links = []
    for raw in parser.metadata:
        url = _web_url(raw, page_url)
        if url and not excluded.search(url) and url not in links:
            links.append(url)
    if links:
        return links[:MAX_CANDIDATES]
    for raw, label in parser.anchors:
        url = _web_url(raw, page_url)
        if not url or excluded.search(raw + " " + label):
            continue
        if urllib.parse.urlsplit(url).hostname != urllib.parse.urlsplit(page_url).hostname:
            continue  # external references to other papers are not copies of this paper
        path = urllib.parse.urlsplit(url).path.lower()
        document = (path.endswith(".pdf") or "/bitstream/" in path or "/bitstreams/" in path or
                    re.search(r"/(?:pdf|epdf|download|document|documents|files)/", path))
        if document and (re.search(r"pdf|download|full.?text|view|manuscript", label, re.I) or "/bitstream" in path):
            if url not in links:
                links.append(url)
    return links[:MAX_CANDIDATES]


def _figshare_article(url):
    """The article number in a figshare article page URL, else None (figshare answers its pages with a bot challenge; its API is open)."""
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or "").lower()
    if host == "api.figshare.com" or not (host == "figshare.com" or host.endswith(".figshare.com")):
        return None
    segments = [s for s in parts.path.split("/") if s]
    if len(segments) < 3 or segments[0] != "articles" or not segments[-1].isdigit():
        return None
    # .../articles/<type>/<title>/<id>[/<version>]
    return segments[-2] if len(segments) >= 4 and segments[-2].isdigit() else segments[-1]


def figshare_pdfs(article, get, agent):
    """(PDF download URLs, why none) from figshare's public API. Embargoed and confidential records are not touched; None means the API itself failed."""
    try:
        reply = get(FIGSHARE + article, max_bytes=MAX_LOOKUP_BYTES, user_agent=agent, accept="application/json")
        record = json.loads(reply.body)
        if reply.status != 200 or not isinstance(record, dict):
            return None, None
    except (FetchError, ValueError):
        return None, None
    if record.get("is_embargoed") or record.get("is_confidential"):
        return [], "figshare record is embargoed or confidential"
    excluded = re.compile(r"supplement|supporting|appendi[xc]|cover.?sheet|terms.of.use", re.I)
    urls = []
    for item in record.get("files") or []:
        name = str(item.get("name") or "")
        if (name.lower().endswith(".pdf") or item.get("mimetype") == "application/pdf") and not item.get("is_link_only") \
                and not excluded.search(name) and item.get("download_url"):
            urls.append(item["download_url"])
    return urls[:MAX_CANDIDATES], None if urls else "figshare record has no PDF file"


def _download_first(links, result, get, agent, *, tried=None, depth=0, landing=False, request_limit=MAX_DOCUMENT_REQUESTS):
    """Try OA documents and up to two HTML hops, preserving the successful copy's provenance."""
    tried = tried if tried is not None else set()

    def fail(**values):
        # A bot-protection wall stays the answer unless another copy downloads: it tells the person to fetch this paper by hand.
        if result["outcome"] != "blocked_by_site":
            result.update(**values)

    for url, version, license in links:
        url = _web_url(url)
        if not url or url in tried or len(tried) >= min(request_limit, MAX_DOCUMENT_REQUESTS):
            continue
        tried.add(url)
        result["pdf_url"] = url
        article = _figshare_article(url) if depth < MAX_PAGE_DEPTH else None
        if article:
            files, why = figshare_pdfs(article, get, agent)
            if files and _download_first([(link, version, license) for link in files], result, get, agent,
                                         tried=tried, depth=depth + 1, request_limit=request_limit):
                return True
            if why:
                fail(outcome="no_pdf_link", detail=why, http_status=None)
                continue
        try:
            reply = get(url, max_bytes=MAX_PDF_BYTES, user_agent=agent, accept="application/pdf,*/*;q=0.5")
        except TooLarge:
            fail(outcome="too_large", detail=url, http_status=None)
            continue
        except FetchError as exc:
            fail(outcome="download_failed", detail=str(exc), http_status=None)
            continue
        if reply.challenge:
            result.update(outcome="blocked_by_site", http_status=reply.status,
                          detail=f"Bot protection at {urllib.parse.urlsplit(reply.url).hostname or 'the site'} (HTTP {reply.status})")
            continue
        if result["outcome"] != "blocked_by_site":
            result["http_status"] = reply.status
        if reply.status != 200:
            fail(outcome="download_failed", detail=f"HTTP {reply.status}")
            continue
        try:
            check_pdf(reply.body)
        except ValueError as exc:
            if not landing:
                fail(outcome="not_a_pdf", detail=f"{exc} ({reply.content_type or 'no content type'})")
            page = _web_url(reply.url) or url
            if depth < MAX_PAGE_DEPTH:
                discovered = page_pdf_links(reply.body, page, result["doi"])
                if discovered:
                    result["landing_url"] = page
                    if _download_first([(link, version, license) for link in discovered], result, get, agent,
                                       tried=tried, depth=depth + 1, request_limit=request_limit):
                        return True
            continue
        result.update(outcome="downloaded", detail=None, data=reply.body, version=version, license=license,
                      pdf_url=_web_url(reply.url) or url)
        if landing:
            result["landing_url"] = url
        return True
    return False


def _try_locations(locations, result, get, agent, tried, *, openalex=False):
    """Try all direct copies first, then their landing pages, including locations with no pdf_url."""
    direct, pages, landing_for = [], [], {}
    request_limit = min(MAX_DOCUMENT_REQUESTS, len(tried) + max(1, MAX_DOCUMENT_REQUESTS // 2))
    for loc in locations:
        loc = loc or {}
        if openalex and not loc.get("is_oa"):
            continue
        version, license = VERSIONS.get(loc.get("version")), loc.get("license")
        pdf = loc.get("pdf_url" if openalex else "url_for_pdf")
        page = loc.get("landing_page_url" if openalex else "url_for_landing_page") or (None if openalex else loc.get("url"))
        landing_for.setdefault(_web_url(pdf), _web_url(page))
        for url, target in ((pdf, direct), (page, pages)):
            url = _web_url(url)
            if url and all(url != item[0] for item in target):
                target.append((url, version, license))
        if page and not result["landing_url"]:
            result["landing_url"] = page
    for link in direct[:MAX_CANDIDATES]:
        if link[0] not in tried:
            result["landing_url"] = landing_for.get(link[0]) or result["landing_url"]
        if _download_first([link], result, get, agent, tried=tried, request_limit=request_limit):
            return True
    if result["outcome"] in (None, "not_open_access"):
        result["outcome"] = "no_pdf_link"
    return _download_first(pages[:MAX_CANDIDATES], result, get, agent, tried=tried, landing=True, request_limit=request_limit)


def _from_unpaywall(doi, email, agent, get):
    result = {"outcome": None, "source": "unpaywall", "found_by": "unpaywall", "doi": doi, "oa_status": None, "license": None, "version": None,
              "pdf_url": None, "landing_url": None, "http_status": None, "detail": None, "data": None}
    try:
        reply = get(UNPAYWALL + urllib.parse.quote(doi, safe="/") + "?" + urllib.parse.urlencode({"email": email}),
                    max_bytes=MAX_LOOKUP_BYTES, user_agent=agent, accept="application/json")
    except FetchError as exc:
        return {**result, "outcome": "lookup_failed", "detail": f"Unpaywall unreachable: {exc}"}, set()
    result["http_status"] = reply.status
    if reply.status == 404:
        return {**result, "outcome": "not_open_access", "detail": "DOI not found in Unpaywall"}, set()
    try:
        record = json.loads(reply.body)
        if reply.status != 200 or not isinstance(record, dict):
            raise ValueError
    except ValueError:
        return {**result, "outcome": "lookup_failed", "detail": f"Unpaywall answered HTTP {reply.status}"}, set()
    best = record.get("best_oa_location") or {}
    result.update(oa_status=record.get("oa_status"), license=best.get("license"), landing_url=best.get("url_for_landing_page") or best.get("url"))
    if not record.get("is_oa"):
        return {**result, "outcome": "not_open_access"}, set()
    tried = set()
    _try_locations([best, *(record.get("oa_locations") or [])], result, get, agent, tried)
    return result, tried


def _from_openalex(doi, result, tried, agent, get, api_key):
    """Ask OpenAlex for other open-access copies of a paper Unpaywall gave no file for. The Unpaywall outcome stands unless OpenAlex adds something."""
    result = {**result, "source": "unpaywall+openalex"}
    query = {"select": OPENALEX_FIELDS, **({"api_key": api_key} if api_key else {})}
    try:
        reply = get(OPENALEX + urllib.parse.quote(doi, safe="/") + "?" + urllib.parse.urlencode(query),
                    max_bytes=MAX_LOOKUP_BYTES, user_agent=agent, accept="application/json")
    except FetchError as exc:
        return {**result, "outcome": "lookup_failed", "detail": f"OpenAlex unreachable: {exc}"}  # not final: a later run asks again
    if reply.status == 404:
        return result  # OpenAlex does not know the DOI; Unpaywall's answer stands
    try:
        record = json.loads(reply.body)
        if reply.status != 200 or not isinstance(record, dict):
            raise ValueError
    except ValueError:
        return {**result, "outcome": "lookup_failed", "detail": f"OpenAlex answered HTTP {reply.status}"}
    open_access = record.get("open_access") or {}
    best = record.get("best_oa_location") or {}
    locations = [best, *(record.get("locations") or [])]
    # best_oa_location and oa_url themselves explicitly identify an open copy.
    if open_access.get("is_oa"):
        locations[0] = {**best, "is_oa": True, "landing_page_url": best.get("landing_page_url") or open_access.get("oa_url")}
    if any(loc and loc.get("is_oa") for loc in locations):
        result.update(found_by="openalex", oa_status=open_access.get("oa_status") or result["oa_status"], landing_url=None)
        _try_locations(locations, result, get, agent, tried, openalex=True)
    return result


def _semantic_scholar_get(get, url, api_key, agent):
    """One Semantic Scholar request at a time for the whole process, SEMANTIC_SCHOLAR_INTERVAL after the previous one ended."""
    global _semantic_scholar_next
    with _semantic_scholar_lock:
        wait = _semantic_scholar_next - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        try:
            return get(url, max_bytes=MAX_LOOKUP_BYTES, user_agent=agent, accept="application/json", headers={"x-api-key": api_key})
        finally:
            _semantic_scholar_next = time.monotonic() + SEMANTIC_SCHOLAR_INTERVAL


def _from_semantic_scholar(doi, result, tried, agent, get, api_key):
    """Ask Semantic Scholar for its open-access PDF of a paper the other sources gave no file for. Their outcome stands unless it adds something."""
    url = SEMANTIC_SCHOLAR + urllib.parse.quote(doi, safe="/") + "?" + urllib.parse.urlencode({"fields": SEMANTIC_SCHOLAR_FIELDS})
    try:
        reply = _semantic_scholar_get(get, url, api_key, agent)
    except FetchError as exc:
        return {**result, "outcome": "lookup_failed", "detail": f"Semantic Scholar unreachable: {exc}"}  # not final: a later run asks again
    if reply.status == 404:
        return result  # Semantic Scholar does not know the DOI
    try:
        record = json.loads(reply.body)
        if reply.status != 200 or not isinstance(record, dict):
            raise ValueError
    except ValueError:
        hint = " (check SEMANTIC_SCHOLAR_API_KEY)" if reply.status in (401, 403) else ""
        return {**result, "outcome": "lookup_failed", "detail": f"Semantic Scholar answered HTTP {reply.status}{hint}"}
    pdf = record.get("openAccessPdf") or {}
    link = _web_url(pdf.get("url"))
    status = str(pdf.get("status") or "").lower() or None
    if not link or link in tried or status == "closed":
        return result
    result = {**result, "found_by": "semantic_scholar", "oa_status": status or result["oa_status"]}
    if result["outcome"] in (None, "not_open_access"):
        result["outcome"] = "no_pdf_link"
    request_limit = min(MAX_DOCUMENT_REQUESTS, len(tried) + max(1, MAX_DOCUMENT_REQUESTS // 2))
    _download_first([(link, None, pdf.get("license"))], result, get, agent, tried=tried, request_limit=request_limit)
    return result


def acquire(doi, email, get=http_get, openalex_key=None, semantic_scholar_key=None):
    """Find a PDF in OA locations and their pages; each discovery service can succeed independently. Semantic Scholar only with a key."""
    doi = _doi(doi)
    if not doi:
        return {"outcome": "no_doi", "source": RESOLVER_VERSION, "found_by": "unpaywall", "doi": doi, "oa_status": None, "license": None, "version": None,
                "pdf_url": None, "landing_url": None, "http_status": None, "detail": None, "data": None}
    agent = f"research0 (mailto:{email})"
    result, tried = _from_unpaywall(doi, email, agent, get)
    lookup_failures = []

    def note(answer):
        if answer["outcome"] == "lookup_failed" and answer["detail"] not in lookup_failures:
            lookup_failures.append(answer["detail"])
        return answer

    note(result)
    if result["outcome"] != "downloaded":
        result = note(_from_openalex(doi, result, tried, agent, get, openalex_key))
    if result["outcome"] != "downloaded" and semantic_scholar_key:
        result = note(_from_semantic_scholar(doi, result, tried, agent, get, semantic_scholar_key))
    # A source that could not be asked leaves the paper open for a later run, whatever the others said.
    if lookup_failures and result["outcome"] != "downloaded":
        result.update(outcome="lookup_failed", detail="; ".join(lookup_failures))
    result["source"] = RESOLVER_VERSION
    return result


# --- Storage ------------------------------------------------------------------------------------------------------------

class S3Store:
    prefix = "fulltext/"

    def __init__(self):
        import boto3
        self.bucket = os.environ["S3_BUCKET"]
        self.client = boto3.client("s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"), region_name=os.getenv("S3_REGION", "us-east-1"))
        self._ready = False

    @property
    def location(self):
        return f"s3://{self.bucket}/{self.prefix}"

    def put(self, key, data, sha256):
        if not self._ready:
            try:
                self.client.head_bucket(Bucket=self.bucket)
            except self.client.exceptions.ClientError as exc:
                if exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 404:
                    raise
                self.client.create_bucket(Bucket=self.bucket)
            self._ready = True
        content_type = CONTENT_TYPES.get(key.rsplit(".", 1)[-1], "application/pdf")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type, Metadata={"sha256": sha256})
        return f"s3://{self.bucket}/{key}"

    def get(self, uri):
        bucket, _, key = uri.removeprefix("s3://").partition("/")
        return self.client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def delete(self, uri):
        bucket, _, key = uri.removeprefix("s3://").partition("/")
        self.client.delete_object(Bucket=bucket, Key=key)


CONTENT_TYPES = {"pdf": "application/pdf", "doc": "application/msword",
                 "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def _digest_and_key(canonical_id, data, extension="pdf"):
    sha = hashlib.sha256(data).hexdigest()
    return sha, f"{S3Store.prefix}{canonical_id}/{sha}.{extension}"


def _insert_file(con, run_id, canonical_id, origin, uri, sha, size, source_url, filename, by, version=None):
    file_id = uuid.uuid4()
    row = con.execute(
        "INSERT INTO fulltext_file(file_id,run_id,canonical_id,origin,s3_uri,sha256,size_bytes,source_url,original_filename,acquired_by,version_label) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (canonical_id,sha256) DO NOTHING RETURNING file_id",
        (file_id, run_id, canonical_id, origin, uri, sha, size, source_url, filename[:200] if filename else None, by, version)).fetchone()
    if row:
        return row[0], True
    return con.execute("SELECT file_id FROM fulltext_file WHERE canonical_id=%s AND sha256=%s", (canonical_id, sha)).fetchone()[0], False


def save_upload(con, store, run_id, canonical_id, data, filename, uploader, *, source_url=None, version=None, allow_word=False):
    """A file chosen by a person: a PDF, or a Word document only when the caller asks for it (the dashboard upload stays PDF-only).
    Same bytes for the same paper are recognised and not stored twice."""
    uploader = (uploader or "").strip()
    if not uploader or len(uploader) > 80:
        raise ValueError("Your name (max 80 characters) is required")
    if allow_word:
        extension = check_document(data)
    else:
        check_pdf(data)
        extension = "pdf"
    if not con.execute("SELECT 1 FROM canonical_paper WHERE run_id=%s AND canonical_id=%s", (run_id, canonical_id)).fetchone():
        raise ValueError("Unknown paper for this catalog run")
    sha, key = _digest_and_key(canonical_id, data, extension)
    uri = store.put(key, data, sha)
    with con.transaction():
        file_id, created = _insert_file(con, run_id, canonical_id, "manual_upload", uri, sha, len(data), source_url, os.path.basename(filename or ""), uploader, version)
    LOG.info(json.dumps({"event": "fulltext_uploaded", "run_id": str(run_id), "canonical_id": str(canonical_id), "sha256": sha, "new": created}))
    return {"file_id": str(file_id), "s3_uri": uri, "sha256": sha, "size_bytes": len(data), "already_stored": not created}


def import_folder(con, store, run_id, folder, uploader):
    """Bulk import: every <canonical_id>.pdf in the folder (the name the missing-papers CSV suggests)."""
    report = {"imported": 0, "already_stored": 0, "rejected": []}
    for path in sorted(Path(folder).glob("*.pdf")):
        try:
            canonical_id = uuid.UUID(path.stem)
        except ValueError:
            report["rejected"].append({"file": path.name, "reason": "file name is not a paper ID (use the suggested_filename column)"})
            continue
        try:
            out = save_upload(con, store, run_id, canonical_id, path.read_bytes(), path.name, uploader)
        except ValueError as exc:
            report["rejected"].append({"file": path.name, "reason": str(exc)})
            continue
        report["already_stored" if out["already_stored"] else "imported"] += 1
    return report


# --- Which papers, what is missing --------------------------------------------------------------------------------------

ELIGIBLE = ("c.run_id=%(run)s AND c.status='retained' AND cardinality(c.perimeter_flags)=0 "
            "AND NOT coalesce((SELECT x.excluded FROM fulltext_exclusion x WHERE x.run_id=c.run_id AND x.canonical_id=c.canonical_id "
            "ORDER BY x.decided_at DESC, x.exclusion_id DESC LIMIT 1), false)")
LATEST = ("SELECT DISTINCT ON (canonical_id) canonical_id,outcome,detail,landing_url,source FROM fulltext_attempt WHERE run_id=%(run)s "
          "ORDER BY canonical_id,attempted_at DESC,attempt_id DESC")
RETRY_SQL = "(l.outcome = ANY(%(retry)s) OR (l.source <> %(resolver)s AND l.outcome = ANY(%(recheck)s)))"


def _selection_params(run_id):
    return {"run": run_id, "retry": list(RETRY), "recheck": list(RECHECK), "resolver": RESOLVER_VERSION}


def parse_years(text):
    """'2023-2025' or '2024' -> (first, last). The proposal limits the first in-depth ingestion to an operational window, 2023-2025 proposed."""
    if text in (None, ""):
        return None
    parts = str(text).split("-")
    try:
        first, last = (int(parts[0]), int(parts[-1])) if len(parts) in (1, 2) else (None, None)
    except ValueError:
        first = last = None
    if first is None or first > last or not 1900 <= first <= 2100:
        raise ValueError("Years must look like 2023-2025")
    return first, last


def papers_in_years(con, run_id, years):
    first, last = years
    return [r[0] for r in con.execute(f"SELECT c.canonical_id FROM canonical_paper c WHERE {ELIGIBLE} AND c.publication_year BETWEEN %(a)s AND %(b)s ORDER BY c.canonical_id",
                                      {"run": run_id, "a": first, "b": last})]


def select_papers(con, run_id, scope, limit=None, years=None, *, include_finished=False):
    """Eligible papers without a file whose last attempt is absent or worth repeating. `years` limits the publication years (the operational window)."""
    if scope not in SCOPES:
        raise ValueError("Scope must be one of: " + ", ".join(SCOPES))
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Limit must be a positive whole number")
    where, params = "", {**_selection_params(run_id), "limit": limit}
    if years:
        where, params["y0"], params["y1"] = " AND c.publication_year BETWEEN %(y0)s AND %(y1)s", years[0], years[1]
    if scope == "sample":
        where += " AND c.canonical_id IN (SELECT canonical_id FROM blind_sample_paper WHERE run_id=%(run)s)"
    elif scope == "candidates":
        # The proposal retrieves full texts for the candidates and the uncertain cases only: papers the latest pre-assignment calls yes or uncertain for either community.
        where += (" AND c.canonical_id IN (SELECT p.canonical_id FROM community_preassignment p WHERE p.label IN ('yes','uncertain') AND p.preassignment_run_id = "
                  "(SELECT preassignment_run_id FROM preassignment_run WHERE run_id=%(run)s AND source <> 'v1_suggestion' ORDER BY created_at DESC LIMIT 1))")
        if not con.execute("SELECT 1 FROM preassignment_run WHERE run_id=%s AND source <> 'v1_suggestion'", (run_id,)).fetchone():
            raise ValueError("No pre-assignment run (yes/no/uncertain) yet: collect it first (the v1 suggestions are not used to choose full texts)")
    elif scope == "audit":
        # Papers in the most recent stratified audit sample: the targeted full text the plan asks for in Phase 4.
        where += ("AND c.canonical_id IN (SELECT p.canonical_id FROM audit_sample_paper p JOIN audit_sample a USING (audit_id) "
                 "WHERE a.run_id=%(run)s AND a.created_at=(SELECT max(created_at) FROM audit_sample WHERE run_id=%(run)s))")
    elif scope == "assigned":
        from .community_dashboard import run_dir
        from .community_recovery import assignment_rows
        # Batch labels are suggestions; here they only decide which papers to fetch first.
        ids = [uuid.UUID(pid) for pid, row in assignment_rows(run_dir(run_id)).items() if row["cto"] == "1" or row["tim"] == "1"]
        if not ids:
            raise ValueError("No collected Batch labels with CTO or TIM yet; use the sample or all eligible papers")
        where, params["ids"] = where + " AND c.canonical_id = ANY(%(ids)s)", ids
    pending = (f"AND (l.outcome IS NULL OR {RETRY_SQL}) "
               "AND NOT EXISTS (SELECT 1 FROM fulltext_file f WHERE f.canonical_id=c.canonical_id) ")
    return con.execute(
        f"WITH latest AS ({LATEST}) SELECT c.canonical_id,coalesce(c.doi,c.all_dois[1]) FROM canonical_paper c "
        f"LEFT JOIN latest l USING (canonical_id) WHERE {ELIGIBLE} {where} "
        + ("" if include_finished else pending) + "ORDER BY c.canonical_id LIMIT %(limit)s", params).fetchall()


def coverage(con, run_id, scope="sample", years=None):
    """Coverage of the entire requested cohort, including unattempted papers (never just this run's successes)."""
    ids = [cid for cid, _ in select_papers(con, run_id, scope, years=years, include_finished=True)]
    have = con.execute("SELECT count(DISTINCT canonical_id) FROM fulltext_file WHERE run_id=%s AND canonical_id=ANY(%s)",
                       (run_id, ids)).fetchone()[0]
    return {"scope": scope, "years": years, "papers": len(ids), "with_file": have,
            "coverage_percent": round(100 * have / len(ids), 2) if ids else None,
            "missing_for_80_percent": max(0, math.ceil(len(ids) * 0.8) - have),
            "missing_for_90_percent": max(0, math.ceil(len(ids) * 0.9) - have)}


def totals(con, run_id):
    row = con.execute(
        f"WITH latest AS ({LATEST}), files AS (SELECT canonical_id,bool_or(origin='open_access_download') AS auto,bool_or(origin='manual_upload') AS manual "
        "FROM fulltext_file WHERE run_id=%(run)s GROUP BY canonical_id) "
        "SELECT count(*), count(f.canonical_id), count(*) FILTER (WHERE f.auto), count(*) FILTER (WHERE f.manual), "
        "count(*) FILTER (WHERE f.canonical_id IS NULL AND l.canonical_id IS NOT NULL AND l.outcome <> 'lookup_failed'), "
        f"count(*) FILTER (WHERE f.canonical_id IS NULL AND {RETRY_SQL}), "
        "count(*) FILTER (WHERE f.canonical_id IS NULL AND l.canonical_id IS NULL) "
        f"FROM canonical_paper c LEFT JOIN latest l USING (canonical_id) LEFT JOIN files f USING (canonical_id) WHERE {ELIGIBLE}", _selection_params(run_id)).fetchone()
    keys = ("eligible", "with_file", "with_open_access_file", "with_uploaded_file", "to_get_by_hand", "to_retry", "not_attempted")
    return dict(zip(keys, row))


MISSING_SQL = (
    f"WITH latest AS ({LATEST}) SELECT c.canonical_id,c.title,c.authors,c.publication_year,c.source_title,coalesce(c.doi,c.all_dois[1]),"
    "l.outcome,l.detail,l.landing_url FROM canonical_paper c JOIN latest l USING (canonical_id) "
    f"WHERE {ELIGIBLE} AND l.outcome NOT IN ('downloaded','lookup_failed') "
    "AND NOT EXISTS (SELECT 1 FROM fulltext_file f WHERE f.canonical_id=c.canonical_id) "
    "ORDER BY c.publication_year DESC NULLS LAST,c.title,c.canonical_id")


def _missing_row(row):
    cid, title, authors, year, source, doi, outcome, detail, landing = row
    return {"id": str(cid), "suggested_filename": f"{cid}.pdf", "title": title, "authors": authors, "year": year, "source": source,
            "doi": doi, "doi_url": f"https://doi.org/{urllib.parse.quote(doi, safe='/()')}" if doi else None,
            "reason_code": outcome, "reason": REASONS.get(outcome, outcome), "detail": detail, "open_access_page": landing}


def missing(con, run_id, offset=0, limit=50):
    """Papers to fetch by hand: attempted, still without a file. Lookup failures are retried, not listed."""
    rows = con.execute(MISSING_SQL + " OFFSET %(offset)s LIMIT %(limit)s", {**_selection_params(run_id), "offset": offset, "limit": limit}).fetchall()
    return [_missing_row(r) for r in rows]


def _cell(value):
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text  # never let a spreadsheet run a title as a formula


def missing_csv(con, run_id):
    out = io.StringIO()
    columns = ("id", "suggested_filename", "doi", "doi_url", "title", "authors", "year", "source", "reason", "detail", "open_access_page")
    writer = csv.writer(out)
    writer.writerow(columns)
    for row in con.execute(MISSING_SQL, _selection_params(run_id)):
        item = _missing_row(row)
        writer.writerow([_cell(item[c]) for c in columns])
    return "﻿" + out.getvalue()


# --- Taking papers out of the full-text work ----------------------------------------------------------------------------

def exclude_papers(con, run_id, canonical_ids, reason, decided_by, excluded=True):
    """Append an exclusion (or, with excluded=False, its reversal): the papers leave the download scopes, the hand-fetch list and the coverage."""
    if not str(reason).strip() or not str(decided_by).strip():
        raise ValueError("An exclusion needs a reason and the name of who decided it")
    with con.transaction():
        for canonical_id in canonical_ids:
            con.execute("INSERT INTO fulltext_exclusion(run_id,canonical_id,excluded,reason,decided_by) VALUES (%s,%s,%s,%s,%s)",
                        (run_id, canonical_id, excluded, reason.strip(), decided_by.strip()))
    LOG.info(json.dumps({"event": "fulltext_exclusion", "run_id": str(run_id), "papers": len(canonical_ids), "excluded": excluded, "decided_by": decided_by}))


def purge_files(con, store, run_id, canonical_ids):
    """Irreversibly delete the stored PDFs of these papers: the S3 object, the file row and the 'downloaded' attempt that points at it.
    The other attempts stay as history. The database goes first, so a storage failure leaves an unreferenced object, never a row without a file."""
    files = con.execute("SELECT file_id,canonical_id,s3_uri FROM fulltext_file WHERE run_id=%s AND canonical_id=ANY(%s) ORDER BY canonical_id,created_at",
                        (run_id, list(canonical_ids))).fetchall()
    ids = [f[0] for f in files]
    with con.transaction():
        attempts = con.execute("DELETE FROM fulltext_attempt WHERE file_id=ANY(%s)", (ids,)).rowcount
        con.execute("DELETE FROM fulltext_file WHERE file_id=ANY(%s)", (ids,))
    failed = []
    for _, _, uri in files:
        try:
            store.delete(uri)
        except Exception as exc:  # noqa: BLE001 - report every object that could not be removed
            failed.append({"s3_uri": uri, "error": str(exc)[:200]})
    LOG.info(json.dumps({"event": "fulltext_purged", "run_id": str(run_id), "files": len(files), "attempts": attempts, "storage_failures": len(failed)}))
    return {"files": [{"file_id": str(f[0]), "canonical_id": str(f[1]), "s3_uri": f[2]} for f in files], "attempts_deleted": attempts, "storage_failures": failed}


# --- Running the downloads ----------------------------------------------------------------------------------------------

def _work(paper, email, get, store, stop):
    if stop.is_set():
        return None
    canonical_id, doi = paper
    result = acquire(doi, email, get, os.environ.get("OPENALEX_API_KEY", "").strip() or None,
                     os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None)
    data = result.pop("data")
    if data:
        result["sha256"], key = _digest_and_key(canonical_id, data)
        result["size"] = len(data)
        result["s3_uri"] = store.put(key, data, result["sha256"])  # a storage failure is systemic: it aborts the run
    return canonical_id, result


def _record(con, run_id, canonical_id, r):
    with con.transaction():
        file_id = None
        if r["outcome"] == "downloaded":
            file_id, _ = _insert_file(con, run_id, canonical_id, "open_access_download", r["s3_uri"], r["sha256"], r["size"], r["pdf_url"], None,
                                      r["found_by"], r["version"])
        con.execute(
            "INSERT INTO fulltext_attempt(run_id,canonical_id,doi,source,outcome,oa_status,license,pdf_url,landing_url,http_status,detail,file_id) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (run_id, canonical_id, r["doi"], r["source"], r["outcome"], r["oa_status"], r["license"], r["pdf_url"], r["landing_url"], r["http_status"],
             r["detail"] and r["detail"][:500], file_id))


def process(con, run_id, papers, email, get=http_get, store=None, workers=WORKERS, progress=None):
    """Try every paper once and append the outcome. Raises if discovery keeps failing or storage is unusable."""
    store = store or S3Store()
    stop, counts, streak = threading.Event(), Counter(), 0
    with ThreadPoolExecutor(workers) as pool:
        try:
            for done in pool.map(lambda p: _work(p, email, get, store, stop), papers):
                if done is None:
                    continue
                canonical_id, r = done
                _record(con, run_id, canonical_id, r)
                counts[r["outcome"]] += 1
                streak = streak + 1 if r["outcome"] == "lookup_failed" else 0
                if progress:
                    progress(dict(counts))
                if streak >= ABORT_AFTER_LOOKUP_FAILURES:
                    raise RuntimeError(f"{streak} lookups failed in a row ({r['detail']}); check UNPAYWALL_EMAIL, SEMANTIC_SCHOLAR_API_KEY and the network")
        except BaseException:
            stop.set()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
    return dict(counts)


class FulltextJob:
    """One background run per server. Progress is in memory; every result is already saved in the database."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"state": "idle"}

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def _set(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, run_id, scope, limit, email, database_url):
        if not email or "@" not in email:
            raise ValueError("Set UNPAYWALL_EMAIL in .env (a real address; Unpaywall asks for it) and restart the server")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 100000):
            raise ValueError("Limit must be a whole number between 1 and 100000")
        import psycopg
        with psycopg.connect(database_url) as con:
            papers = select_papers(con, run_id, scope, limit)
        with self.lock:
            if self.state["state"] == "running":
                raise ValueError("A download is already running")
            self.state = {"state": "running", "scope": scope, "limit": limit, "total": len(papers), "done": 0, "outcomes": {},
                          "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "error": None}
        threading.Thread(target=self._run, args=(run_id, papers, email, database_url), daemon=True).start()
        return self.snapshot()

    def _run(self, run_id, papers, email, database_url):
        import psycopg
        try:
            with psycopg.connect(database_url) as con:
                process(con, run_id, papers, email, progress=lambda c: self._set(done=sum(c.values()), outcomes=c))
                self._set(coverage=coverage(con, run_id, self.snapshot()["scope"]))
            self._set(state="finished")
        except Exception as exc:  # reported to the page; results saved so far stay saved
            LOG.error(json.dumps({"event": "fulltext_job_failed", "run_id": str(run_id), "error": str(exc)[:300]}))
            self._set(state="failed", error=str(exc)[:300])
        finally:
            self._set(finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
