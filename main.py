#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEXC Futures Arbitrage Bot - Main Entry Point
A sophisticated trading bot leveraging MEXC Futures vs DEX arbitrage opportunities
"""

import sys
import logging
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "bot.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'PyQt5': 'PyQt5',
        'requests': 'requests',
        'ccxt': 'ccxt',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'python-dotenv': 'dotenv'
    }
    
    missing = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    logger.info("All dependencies are installed ✓")
    return True

def main():
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("MEXC Futures Arbitrage Bot Starting...")
    logger.info("=" * 60)
    
    if not check_dependencies():
        sys.exit(1)
    
    try:
        from gui.main_window import BotMainWindow
        
        app = QApplication(sys.argv)
        window = BotMainWindow()
        window.show()
        
        logger.info("GUI Window initialized successfully")
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.exception(f"Fatal error during initialization: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()