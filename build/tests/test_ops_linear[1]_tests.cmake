add_test([=[LinearOpTest.ForwardPass2x2]=]  /workspace/build/tests/test_ops_linear [==[--gtest_filter=LinearOpTest.ForwardPass2x2]==] --gtest_also_run_disabled_tests)
set_tests_properties([=[LinearOpTest.ForwardPass2x2]=]  PROPERTIES WORKING_DIRECTORY /workspace/build/tests SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set(  test_ops_linear_TESTS LinearOpTest.ForwardPass2x2)
