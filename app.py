"""
EMMA - Experimental Music Making Algorithm
Main application entry point

Copyright (c) 2025 Gamahea / LEMM Project
Licensed under Apache 2.0
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.settings import settings
from src.utils.logger import setup_logger
from src.utils.device_utils import get_device_info
from src.ui.gradio_app import EmmaUI


def main():
    """Main entry point for EMMA"""
    
    # Setup logging
    logger = setup_logger(
        name="emma",
        level=logging.INFO,
        log_file="logs/emma.log",
        console=True
    )
    
    logger.info("=" * 60)
    logger.info("EMMA - Experimental Music Making Algorithm")
    logger.info("Gamahea / LEMM Project")
    logger.info("=" * 60)
    
    # Display device information
    device_info = get_device_info()
    logger.info("Device Information:")
    for key, value in device_info.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize and launch UI
    ui = None
    try:
        logger.info("Initializing EMMA...")
        
        ui = EmmaUI()
        ui.initialize()
        
        logger.info("Starting Gradio UI...")
        ui.launch(
            share=settings.ui.share,
            server_port=settings.ui.server_port
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down EMMA...")
        if ui:
            ui.cleanup()
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error running EMMA: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
