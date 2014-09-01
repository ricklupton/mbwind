USERCONFIG=tests/regression/userconfig
JOBCONFIG=tests/regression/jobconfig
TESTCODE_ARGS=--userconfig=${USERCONFIG} --jobconfig=${JOBCONFIG}

regression-test:
	~/src/testcode/bin/testcode.py ${TESTCODE_ARGS}

unit-test:
	nosetests --with-coverage --cover-package mbwind \
	  --cover-html --cover-html-dir=cover

test: unit-test regression-test

clean:
	~/src/testcode/bin/testcode.py ${TESTCODE_ARGS} tidy
