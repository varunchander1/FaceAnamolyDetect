"""Every environment variable the app reads, in one validated place.

Field names map to their upper-case env var (`vlm_model` <- `VLM_MODEL`), and a
`.env` file in the working directory is picked up automatically — it is
gitignored, because it is where the API key normally lives.

Import `settings` and read attributes off it. Nothing else in the codebase
should touch `os.environ`, so the full configuration surface is this file.

The key is a `SecretStr`: printing, logging or repr-ing the settings object
shows `**********` instead of the value. Call `require_api_key()` to get the
plain string at the point of use.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_GALLERY_PATH = "gallery.npz"
DEFAULT_FACES_DIR = "faces"
DEFAULT_TRAINING_IMAGES_DIR = "labeldata/training"
API_KEY_ENV = "OPENAI_API_KEY"


class Settings(BaseSettings):
    """Runtime configuration, read once from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # the environment is full of unrelated variables
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        description="OpenAI key for scene descriptions. Detection needs no key.",
    )
    vlm_model: str = Field(
        default=DEFAULT_MODEL,
        description="Any OpenAI model with image input.",
    )
    vlm_fallback: str = Field(
        default=DEFAULT_MODEL,
        description="Comma-separated models tried when an answer comes back empty.",
    )
    gallery_path: str = Field(
        default=DEFAULT_GALLERY_PATH,
        description="Enrolled face embeddings, written by the Train dialog.",
    )
    faces_dir: str = Field(
        default=DEFAULT_FACES_DIR,
        description="Reference photos, one folder per person.",
    )
    training_images_dir: str = Field(
        default=DEFAULT_TRAINING_IMAGES_DIR,
        description=(
            "Source images for bulk enrolment. Label Studio exports carry "
            "labels only, so the pixels are read from here."
        ),
    )

    @field_validator("vlm_model", "gallery_path", "faces_dir", "training_images_dir")
    @classmethod
    def _not_blank(cls, value: str, info) -> str:
        """An empty override is a mistake, not a request for empty-string."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(f"{info.field_name.upper()} must not be blank")
        return cleaned

    @property
    def fallback_chain(self) -> tuple[str, ...]:
        """`VLM_FALLBACK` split into clean model names."""
        return tuple(name.strip() for name in self.vlm_fallback.split(",") if name.strip())

    @property
    def has_api_key(self) -> bool:
        """Whether a key is configured. Never returns the key itself."""
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    def require_api_key(self) -> str:
        """The key, or a ValueError naming the variable to set."""
        if not self.has_api_key:
            raise ValueError(
                f"{API_KEY_ENV} is not set. Export your OpenAI API key before "
                f"starting:  export {API_KEY_ENV}=sk-...  "
                "(or put it in a .env file)"
            )
        assert self.openai_api_key is not None  # narrowed by has_api_key
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings, read once."""
    return Settings()


settings = get_settings()
