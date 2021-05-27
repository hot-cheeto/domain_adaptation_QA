import logging
import sys
import os
from logging.handlers import RotatingFileHandler

# Default log output
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s.%(funcName)s : %(message)s', 
                              datefmt = '%m/%d/%Y %I:%M:%S %p')


def get_streamHandler(level = 'INFO', 
                      formats = formatter):
    """
    Note:
        Creates a console handler to display 
        program output
    @parameter:
        level is string value what to disply on the log
        formats regex string what format should i display it 
    @return:
        stream handler object

    """
    ch = logging.StreamHandler()
    ch.setFormatter(formats)
    ch.setLevel(level)
    
    return ch


def get_fileHandler(filename = 'experiment.log', 
                    level = 'INFO', 
                    formats = formatter, 
                    maxBytes = 20, 
                    backupCount = 5):
    """
    Note: 
        creates a file handler to display program output
    @parameter:
        filename of the log should write to 
        level the among of information display on the file
        maxbytes how big should one file should be 
        backupCount is file exceds maxbytes how many files 
        should be create.
    @return:
        file handler object

    """
    fh=RotatingFileHandler(filename, maxBytes, backupCount)
    fh.setFormatter(formats)
    fh.setLevel(level)
    
    return fh


def create_logger(name = '__name__', 
                  level = 'INFO'):
    """
    Note:
        creates logs with a list of handlers
    @paramter:
        name is name where the log first starts
        level how much infomation should be disply 
    @return:
        logger object

    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    
    return logger


if __name__ == '__main__':
    create_logger()

