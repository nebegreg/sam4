"""
Système de logging centralisé pour SAM3 Roto
"""
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration du logging
DEBUG_MODE = True  # À mettre False pour désactiver les logs verbeux

class ColoredFormatter(logging.Formatter):
    """Formatter avec couleurs pour la console"""

    # Codes couleur ANSI
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Vert
        'WARNING': '\033[33m',   # Jaune
        'ERROR': '\033[31m',     # Rouge
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        # Ajouter le nom du thread
        record.threadName = threading.current_thread().name

        # Colorier le niveau
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


class SAM3Logger:
    """Logger centralisé pour SAM3 Roto"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._initialized = True
        self.loggers = {}

        # Créer le dossier de logs
        self.log_dir = Path.home() / ".sam3roto" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Fichier de log avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"sam3roto_{timestamp}.log"

        # Garder seulement les 10 derniers logs
        self._cleanup_old_logs()

        # Logger principal
        self.main_logger = self._create_logger("SAM3Roto", self.log_file)

        print(f"[LOGGING] Logs sauvegardés dans: {self.log_file}")

    def _cleanup_old_logs(self):
        """Garde seulement les 10 derniers fichiers de log"""
        log_files = sorted(self.log_dir.glob("sam3roto_*.log"), key=lambda p: p.stat().st_mtime)
        for old_log in log_files[:-10]:
            try:
                old_log.unlink()
            except Exception:
                pass

    def _create_logger(self, name: str, log_file: Path) -> logging.Logger:
        """Crée un logger configuré"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
        logger.propagate = False

        # Éviter les doublons de handlers
        if logger.handlers:
            return logger

        # Handler fichier (tous les niveaux)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(threadName)-15s | %(name)-20s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Handler console (INFO et plus)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
        console_formatter = ColoredFormatter(
            '[%(threadName)s] %(name)s | %(levelname)s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        return logger

    def get_logger(self, name: str) -> logging.Logger:
        """Obtient ou crée un logger pour un module"""
        if name not in self.loggers:
            self.loggers[name] = self._create_logger(name, self.log_file)
        return self.loggers[name]

    def log_exception(self, logger_name: str, message: str, exc_info=True):
        """Log une exception avec stack trace complète"""
        logger = self.get_logger(logger_name)
        logger.error(message, exc_info=exc_info)

    def set_debug_mode(self, enabled: bool):
        """Active ou désactive le mode debug"""
        global DEBUG_MODE
        DEBUG_MODE = enabled

        level = logging.DEBUG if enabled else logging.INFO
        for logger in self.loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(level)


# Instance globale
_logger_instance = SAM3Logger()

def get_logger(name: str) -> logging.Logger:
    """Obtient un logger pour un module"""
    return _logger_instance.get_logger(name)

def log_exception(logger_name: str, message: str, exc_info=True):
    """Log une exception avec stack trace"""
    _logger_instance.log_exception(logger_name, message, exc_info)

def set_debug_mode(enabled: bool):
    """Active ou désactive le mode debug"""
    _logger_instance.set_debug_mode(enabled)

def get_log_file() -> Path:
    """Retourne le chemin du fichier de log actuel"""
    return _logger_instance.log_file


# Logger par défaut pour les imports
logger = get_logger("SAM3Roto")
