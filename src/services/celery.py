from celery import Celery
from src.config.appConfig import appConfig
from src.rag.ingestion.ingestion import process_document

celery_app = Celery(
    "multi-modal-rag",  # Celery 앱 이름
    broker=appConfig["redis_url"],  # broker - Redis Queue - 태스크가 큐에 저장됨
)


@celery_app.task
def perform_rag_ingestion_task(document_id: str):
    try:
        process_document_result = process_document(document_id)
        return (
            f"Document {process_document_result['document_id']} processed successfully"
        )
    except Exception as e:
        return f"Failed to process document {document_id}: {str(e)}"