# for file in find(); do
#     unit = $(basename $file .py)
#     make unit_test UNIT_TEST=$unit
# done

for file in $(find tests/python -name *.py); do
    unit=$(basename $file .py)
    make unit_test UNIT_TEST=$unit
done
