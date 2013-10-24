
COVER_OPTIONS = '--with-coverage --cover-package mbwind --cover-html --cover-html-dir=../cover'
NOSE_OPTIONS = '%s' % COVER_OPTIONS

def task_test():
    return {
        'actions': ['spec %s' % NOSE_OPTIONS]
    }
