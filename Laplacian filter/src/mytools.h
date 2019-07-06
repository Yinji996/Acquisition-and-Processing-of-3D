#ifndef MYTOOLS
#define MYTOOLS

#define NOMINMAX

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <igl/vertex_triangle_adjacency.h>
#include <random>
#include <Eigen/LU>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/setdiff.h> 
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/adjacency_list.h>
#include <igl/eigs.h>
#define M_PI 3.141592653589793238462643383279

#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Eigen/SparseCore>




void get_boundary_vex(Eigen::MatrixXd const & V_in, Eigen::MatrixXi const & F_in, Eigen::MatrixXd & out_bvex);
void get_boundary_edges(Eigen::MatrixXi const & F_in, Eigen::MatrixXi & out_bedge);

void compute_H(Eigen::MatrixXd const & V, Eigen::MatrixXi const & F, Eigen::VectorXd & H);

void calculate_vertex_normal(
	Eigen::MatrixXd const & V, 
	Eigen::MatrixXi const & F, 
	Eigen::MatrixXd const & FN,
	Eigen::MatrixXd & out_VN);

Eigen::MatrixXd uniform_mean_curvature(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXi const & m_F,
	Eigen::MatrixXd const & m_VN);

Eigen::MatrixXd uniform_gaussian_curvature(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXi const & m_F);

Eigen::MatrixXd discrete_mean_curvature(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXi const & m_F,
	Eigen::MatrixXd const & m_VN);

Eigen::MatrixXd mesh_reconstruction(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXi const & m_F,
	Eigen::MatrixXd const & m_VN,
    int const & k);

Eigen::MatrixXd explicit_Smoothing(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXi const & m_F,
	double const & lamda_1);

Eigen::MatrixXd implicit_Smoothing(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXi const & m_F,
	double const & lamda_2);

float Error_tester(
	Eigen::MatrixXd const & m_V,
	Eigen::MatrixXd const & m_V_new);

#endif


