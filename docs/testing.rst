Testing
=======

Before making a pull request, if possible provide tests for added features or bug fixes.

We have an automated building system which also runs all of provided tests. In case any of the test cases fails, we are notified about failing tests. Those should be fixed before we merge your pull request to master branch.

For the purpose of checking if all test are passing localy you can run following command:

.. code:: sh

    $ poetry run coverage run --source=niaaml -m unittest discover -b

If all tests passed running this command it is most likely that the tests would pass on our build system too.