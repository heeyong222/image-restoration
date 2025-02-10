from loguru import logger

class TqdmLoggingHandler:
    def write(self, message):
        message = message.strip()
        if message:
            logger.debug(message)
    
    def flush(self):
        pass