import logging

import coloredlogs


def logger_config(logger, logger_name: str = None):
    """

    :param logger:
    :param logger_name:
    :return:
    """
    # console handler for print logs
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # formatter for the handlers
    formatter = logging.Formatter(fmt='%(funcName)s:%(lineno)d\n%(message)s')
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)

    if logger_name is not None:
        # file handler for save logs
        fh = logging.FileHandler(logger_name, mode='a')
        fh.setLevel(logger.getEffectiveLevel())
        formatter = logging.Formatter(fmt='%(funcName)s:%(lineno)d\n%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    encoded_styles = 'info=green;warning=yellow;critical=red,bold,exception=blue'
    coloredlogs.install(level=logging.INFO, fmt='%(funcName)s:%(lineno)d\n%(message)s',
                        level_styles=coloredlogs.parse_encoded_styles(encoded_styles), logger=logger)
    return logger
