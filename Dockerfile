FROM python:3.13       # Base image

WORKDIR /app          

RUN apt-get update && apt-get install -y \          # 시스템 의존성 설치 
	poppler-utils \
	tesseract-ocr \
	libmagic-dev  \
	libgl1 \
	libglib2.0-0 \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*                # 캐시 파일 삭제 
	
RUN pip install --no-cache-dir poetry             # 의존성 관리 도구 설치 

COPY pyproject.toml poetry.lock* ./           

RUN poetry config virtualenvs.create false       

RUN poetry install --no-interaction --no-ansi --no-root  # 파이썬 의존성 설치

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]