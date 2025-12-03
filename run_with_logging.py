#!/usr/bin/env python3
"""
Script de démarrage SAM3 Roto avec logging et capture d'erreurs
"""
import sys
import os
import signal
import traceback
from pathlib import Path

# Ajouter le module au path
sys.path.insert(0, str(Path(__file__).parent))

def signal_handler(signum, frame):
    """Capture les signaux (SIGSEGV, SIGABRT, etc.)"""
    print(f"\n{'='*80}")
    print(f"SIGNAL CAPTURÉ: {signal.Signals(signum).name}")
    print(f"{'='*80}")
    print("Stack trace:")
    traceback.print_stack(frame)
    print(f"{'='*80}\n")

    # Essayer de sauvegarder un rapport
    try:
        from sam3roto.utils.logging import get_log_file
        log_file = get_log_file()
        print(f"Logs sauvegardés dans: {log_file}")
    except Exception:
        pass

    sys.exit(1)

def main():
    """Point d'entrée principal avec gestion d'erreurs"""

    # Configurer les handlers de signaux
    # Note: SIGSEGV ne peut pas toujours être capturé, mais on essaie
    try:
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        # signal.signal(signal.SIGSEGV, signal_handler)  # Dangereux, désactivé
    except Exception as e:
        print(f"Warning: Could not set signal handlers: {e}")

    print("="*80)
    print("SAM3 Roto Ultimate - Démarrage avec logging activé")
    print("="*80)

    # Importer et configurer le logging
    try:
        from sam3roto.utils.logging import get_logger, get_log_file, set_debug_mode
        set_debug_mode(True)  # Active le mode debug
        logger = get_logger("Main")
        log_file = get_log_file()

        print(f"✓ Logging configuré: {log_file}")
        logger.info("="*60)
        logger.info("SAM3 Roto Ultimate - Démarrage")
        logger.info("="*60)
    except Exception as e:
        print(f"Warning: Could not initialize logging: {e}")
        import logging
        logger = logging.getLogger("Main")
        logger.setLevel(logging.INFO)

    # Importer Qt avant tout
    try:
        logger.info("Import de PySide6...")
        from PySide6 import QtWidgets, QtCore
        logger.info("✓ PySide6 importé")

        # Configurer Qt pour être plus verbeux
        os.environ['QT_LOGGING_RULES'] = '*.debug=true'
        os.environ['QT_DEBUG_PLUGINS'] = '1'

    except ImportError as e:
        logger.error(f"Erreur import PySide6: {e}", exc_info=True)
        print("ERREUR: PySide6 non installé")
        print("Installation: pip install PySide6")
        return 1

    # Créer l'application Qt
    try:
        logger.info("Création QApplication...")
        app = QtWidgets.QApplication(sys.argv)
        logger.info("✓ QApplication créée")

        # Configurer Qt pour gérer les erreurs de threading
        QtCore.QObject.connect(
            app,
            QtCore.SIGNAL("aboutToQuit()"),
            lambda: logger.info("Application Qt en cours de fermeture...")
        )

    except Exception as e:
        logger.error(f"Erreur création QApplication: {e}", exc_info=True)
        print(f"ERREUR: {e}")
        return 1

    # Importer et créer la fenêtre principale
    try:
        logger.info("Import de l'application principale...")
        from sam3roto.app import MainWindow
        logger.info("✓ Application importée")

        logger.info("Création de la fenêtre principale...")
        window = MainWindow()
        logger.info("✓ Fenêtre principale créée")

        logger.info("Affichage de la fenêtre...")
        window.show()
        logger.info("✓ Fenêtre affichée")

    except Exception as e:
        logger.error(f"Erreur création fenêtre: {e}", exc_info=True)
        print(f"ERREUR: {e}")
        traceback.print_exc()
        return 1

    # Lancer la boucle événementielle
    try:
        logger.info("Démarrage de la boucle événementielle Qt...")
        logger.info("="*60)
        print("\n✓ Application démarrée. Consultez les logs pour les détails.\n")

        ret = app.exec()

        logger.info("="*60)
        logger.info(f"Application fermée avec code: {ret}")
        return ret

    except Exception as e:
        logger.error(f"Erreur dans la boucle principale: {e}", exc_info=True)
        print(f"ERREUR: {e}")
        traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        logger.info("Interruption clavier (Ctrl+C)")
        print("\nInterruption utilisateur")
        return 0
    finally:
        logger.info("Nettoyage final...")
        try:
            from sam3roto.utils.logging import get_log_file
            print(f"\n{'='*80}")
            print(f"Logs sauvegardés dans: {get_log_file()}")
            print(f"{'='*80}\n")
        except Exception:
            pass

if __name__ == "__main__":
    sys.exit(main())
