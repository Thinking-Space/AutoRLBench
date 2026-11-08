"""Select the configured retrieval endpoint for validation."""
def validate(trainer):
    original = trainer.config.retriever.url
    try:
        trainer.config.retriever.url = trainer.config.retriever.validation_url
        return trainer._validate_once()
    finally:
        trainer.config.retriever.url = original
