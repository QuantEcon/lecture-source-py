echo "PR: $TRAVIS_PULL_REQUEST"
echo "COMMIT RANGE: $TRAVIS_COMMIT_RANGE"
CHANGED_FILES=$(git diff --name-only $TRAVIS_COMMIT_RANGE | grep '\.rst')
echo "List of Changed Files: $CHANGED_FILES"
if [ -z "$CHANGED_FILES"]; then
    echo "No RST Files have changed -- nothing to do in this PR"
else
    make coverage FILES=$CHANGED_FILES
    more _build/coverage/jupyter/reports/code-execution-results.json
fi
