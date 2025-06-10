configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TTL_TEST_DEPENDS
        FileCheck count not
        ttl-cc ttl-opt
        )

add_lit_testsuite(check-ttl "Running the TTL regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TTL_TEST_DEPENDS}
        )
set_target_properties(check-ttl PROPERTIES FOLDER "Tests")

add_lit_testsuites(TTL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TTL_TEST_DEPENDS})

