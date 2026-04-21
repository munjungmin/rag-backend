from urllib.parse import urlparse


def validate_url(url_string: str) -> bool:
    if not isinstance(url_string, str) or not url_string.strip():
        return False

    try:
        parsed_url = urlparse(url_string)
        return bool(parsed_url.scheme) and bool(parsed_url.netloc)
    except Exception:
        # 잘못된 형식의 URL 파싱 오류 처리
        return False