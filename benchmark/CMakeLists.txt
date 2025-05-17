add_cuda_executable(iobench FALSE iobench.cc)
target_link_libraries(iobench PUBLIC hshm::cudacxx MPI::MPI_CXX cufile)

install(TARGETS
    iobench
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)