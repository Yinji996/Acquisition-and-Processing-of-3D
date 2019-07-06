#include "mytools.h"


void get_boundary_vex(
	Eigen::MatrixXd const & V, 
	Eigen::MatrixXi const & F,
	Eigen::MatrixXd & out_bvex)
{
	// You job is to use igl::boundary_loop to find boundary vertices
	// 
	// V : input vertices, N-by-3
	// F : input faces
	//
	// out_bvex : output vertices, K-by-3
	// 
	//  Hints:
	//   Eigen::VectorXi b_bex_index
	//     igl::boundary_loop( F_in , b_bex_index )
	// 

	Eigen::VectorXi b;
	igl::boundary_loop(F, b);
	// List of all vertex indices
	Eigen::VectorXi all, in;
	igl::colon<int>(0, V.rows() - 1, all);

	// List of interior indices
	Eigen::VectorXi IA;
	igl::setdiff(all, b, in, IA);


	out_bvex.resize(b.rows(), 3);
	for (size_t i = 0; i < b.rows(); i++)
	{
		out_bvex.row(i) = V.row(b[i]);
	}

}

void get_boundary_edges( 
	Eigen::MatrixXi const & F,
	Eigen::MatrixXi & out_b_edge)
{
	// You job is to use igl::boundary_facets to find boundary edges
	//  
	// F : input faces
	//
	// out_bedge : output edges, K-by-2 (two vertices index)
	// 
	//  Hints:
	//   Eigen::MatrixXi b_edge
	//     igl::boundary_facets( F_in , b_edge )
	//  

	// Find boundary edges
	Eigen::MatrixXi E;
	igl::boundary_facets(F, E);

	// Find boundary vertices
	//Eigen::VectorXi b, IA, IC;
	//igl::unique(E, b, IA, IC);

	//std::cout << "E=" << E.rows() << "," << E.cols() << "\n";

	out_b_edge = E;

}

void compute_H(
	Eigen::MatrixXd const & V,
	Eigen::MatrixXi const & F,
	Eigen::VectorXd & H
	)
{
	// You job is to use igl::cotmatrix, igl::massmatrix, igl::invert_diag to compute mean curvature H at each vertex
	// 
	// V : input vertices, N-by-3
	// F : input faces
	//
	// H : output vertices, K-by-1
	// 
	//
	// Hints
	// Compute Laplace-Beltrami operator
	//	Eigen::SparseMatrix<double> L, Area, AreaInv;
	//	igl::cotmatrix(V, F, L);
	//	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, Area );
	//	

	//------------------------------------------
	// replace this 
	H.resize(V.rows());
	H.setZero();
	//------------------------------------------
	Eigen::SparseMatrix<double> L, Area, AreaInv;
	igl::cotmatrix(V, F, L);
	igl::massmatrix(V, F, igl::MASSMATRIX_TYPE_VORONOI, Area);

	Eigen::MatrixXd HN;
	igl::invert_diag(Area, AreaInv);
	HN = -0.5*AreaInv * (L*V);
	H = HN.rowwise().norm(); //up to sign
	 
}


void calculate_vertex_normal(Eigen::MatrixXd const & V, Eigen::MatrixXi const & F, Eigen::MatrixXd const & FN, Eigen::MatrixXd & out_VN)
{
	//
	// input:
	//   V: vertices
	//   F: face 
	//   FN: face normals
	// output:
	//   out_VN
	//
	//   Your job is to implement vertex normal calculation
	//

	std::vector<std::vector<int> > VF;
	std::vector<std::vector<int> > VFi;
	igl::vertex_triangle_adjacency(V.rows(), F, VF, VFi);


	Eigen::MatrixXd VN(V.rows(), 3);

	for (int i = 0; i < V.rows(); i++)
	{
		Eigen::RowVector3d nv(0, 0, 0);

		for (int j = 0; j < VF[i].size(); j++)
		{
			nv = nv + FN.row(VF[i][j]);
		}

		nv = nv / VF[i].size();

		VN.row(i) = nv;
	}

	out_VN = VN;
}


/****************compute Area******************/
double computeArea(Eigen::Vector3d &V1, Eigen::Vector3d &V2, Eigen::Vector3d &V3) {
	Eigen::Vector3d vec1 = V1 - V2;
	Eigen::Vector3d vec2 = V1 - V3;
	Eigen::Vector3d crossValue(3,1);
	crossValue = vec1.cross(vec2);
	double area_value = crossValue.norm() /2;
	return area_value;
}

/****************compute Angle******************/
double computeAngle(Eigen::Vector3d &V1, Eigen::Vector3d &V2, Eigen::Vector3d &V3) {
	Eigen::Vector3d vec1 = V2 - V1;
	Eigen::Vector3d vec2 = V3 - V1;

	double norm1 = vec1.norm();
	double norm2 = vec2.norm();

	double angle = acos(vec1.dot(vec2) / (norm1*norm2));
	return angle;
}


/****************compute Contan******************/
double computeContan(Eigen::Vector3d &V1, Eigen::Vector3d &V2, Eigen::Vector3d &V3) {
	Eigen::Vector3d vec1 = V2 - V1;
	Eigen::Vector3d vec2 = V3 - V1;

	double norm1 = vec1.norm();
	double norm2 = vec2.norm();

	double angle = acos(vec1.dot(vec2) / (norm1*norm2));
	double cotan = 1 / tan(angle);
	return cotan;
}


/*********************uniform_mean_curvature**************************/
Eigen::MatrixXd uniform_mean_curvature(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F, Eigen::MatrixXd const & m_VN)
{
	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex);

	Eigen::SparseMatrix<double> Delta(m_V.rows(), m_V.rows());//build sparse matrix
	int num; //obtain adjacent number
	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		Delta.coeffRef(i, i) = 1;
		num = adjacency_vertex[i].size();
		int place;
		for (int j = 0; j < num; j++)
		{
			place = adjacency_vertex[i][j];
			Delta.coeffRef(i, place) = -1 / double(num);
		}
	}
	//std::cout << Delta;
	Eigen::MatrixXd DelF(m_V.rows(), m_V.cols()); //delta product F
	DelF = Delta * m_V;
	//std::cout << DelF;
	Eigen::MatrixXd UniformMean(m_V.rows(), 1);
	UniformMean = DelF.rowwise().norm();
	//UniformMean = DelF / (2 * m_VN);
	UniformMean = UniformMean / 2.0;
	//std::cout << UniformMean;
	Eigen::MatrixXd signCurv(m_V.rows(), 1); //find sign of mean curature
	signCurv = (DelF.cwiseProduct(m_VN)).rowwise().sum(); // compute the dot product 
	signCurv = signCurv.array() / signCurv.array().abs().array(); //judge sign

	UniformMean = UniformMean.cwiseProduct(-1.f*signCurv);

	return UniformMean;
}


/*********************uniform_gaussian_curvature**************************/
Eigen::MatrixXd uniform_gaussian_curvature(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F)
{
	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex,true);
	int num; //obtain the number of adjacent
	Eigen::Vector3d V1, V2, V3;
	Eigen::MatrixXd gaussianCurv(m_V.rows(), 1);
	double currentArea, currentAngle, sumArea, sumAngle; // calculate current area and angle
	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sumArea = 0.0;
		sumAngle = 0.0;
		V1 = m_V.row(i);
		num = adjacency_vertex[i].size();
		for (int j = 0; j < num - 1; j++)
		{

			V2 = m_V.row(adjacency_vertex[i][j]);
			V3 = m_V.row(adjacency_vertex[i][j+1]);

			// compute the angle for the triangle and the area 
			currentArea = computeArea(V1, V2, V3);
			currentAngle = computeAngle(V1, V2, V3);

			sumArea += currentArea;
			sumAngle += currentAngle;
			
		}
		V2 = m_V.row(adjacency_vertex[i][0]);
		V3 = m_V.row(adjacency_vertex[i][num - 1]);

		// compute the angle for the triangle and the area 
		currentArea = computeArea(V1, V2, V3);
		currentAngle = computeAngle(V1, V2, V3);

		sumArea += currentArea;
		sumAngle += currentAngle;
		//std::cout << (2.0*M_PI - sumAngle)  << std::endl;
		gaussianCurv(i, 0) = (2.0* M_PI - sumAngle) / (sumArea / 3.0);		
	}
	//std::cout << sumAngle << std::endl;
	return gaussianCurv;
}

/*********************discrete_mean_curvature**************************/
Eigen::MatrixXd discrete_mean_curvature(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F, Eigen::MatrixXd const & m_VN)
{
	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex,true);

	Eigen::SparseMatrix<double> M_Inverse(m_V.rows(), m_V.rows());//build Sparse matrix of M
	int num; //obtain adjacent number
	double sumArea, currentArea;
	Eigen::Vector3d V1, V2, V3, V4;


	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sumArea = 0.0;
		V1 = m_V.row(i);
		num = adjacency_vertex[i].size();
		for (int j = 0; j < num - 1; j++)
		{
			V2 = m_V.row(adjacency_vertex[i][j]);
			V3 = m_V.row(adjacency_vertex[i][j + 1]);

			// compute the angle for the triangle and the area 
			currentArea = computeArea(V1, V2, V3);
			sumArea += currentArea;
		}
		V2 = m_V.row(adjacency_vertex[i][0]);
		V3 = m_V.row(adjacency_vertex[i][num - 1]);

		// compute the angle for the triangle and the area 
		currentArea = computeArea(V1, V2, V3);
		sumArea += currentArea;

		M_Inverse.coeffRef(i, i) = 1 / (sumArea/3*2); // obtain M^(-1)
	}

	Eigen::SparseMatrix<double> Cotan_Weight(m_V.rows(), m_V.rows());//Build Sparse matrix of C
	double alpha, beta;
	double sum_alpha_beta;
	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sum_alpha_beta = 0;
		num = adjacency_vertex[i].size();
		int place;
		for (int j = 0; j < num; j++)
		{
			place = adjacency_vertex[i][j];
			if (j == 0)
			{
				V1 = m_V.row(adjacency_vertex[i][num - 1]);
				V2 = m_V.row(i);
				V3 = m_V.row(adjacency_vertex[i][j]);
				V4 = m_V.row(adjacency_vertex[i][j+1]);
				alpha = computeContan(V1, V2, V3);
				beta = computeContan(V4, V2, V3);
			}
			else if (j == num - 1)
			{
				V1 = m_V.row(adjacency_vertex[i][j - 1]);
				V2 = m_V.row(i);
				V3 = m_V.row(adjacency_vertex[i][j]);
				V4 = m_V.row(adjacency_vertex[i][0]);
				alpha = computeContan(V1, V2, V3);
				beta = computeContan(V4, V2, V3);
			}
			else
			{
				V1 = m_V.row(adjacency_vertex[i][j - 1]);
				V2 = m_V.row(i);
				V3 = m_V.row(adjacency_vertex[i][j]);
				V4 = m_V.row(adjacency_vertex[i][j + 1]);
				alpha = computeContan(V1, V2, V3);
				beta = computeContan(V4, V2, V3);
			}		
			Cotan_Weight.coeffRef(i, place) = alpha + beta;
			sum_alpha_beta = sum_alpha_beta + alpha + beta;
		}
		Cotan_Weight.coeffRef(i, i) = - sum_alpha_beta;
	}

	Eigen::MatrixXd DelF(m_V.rows(), m_V.cols()); //delta product F
	DelF = M_Inverse * Cotan_Weight * m_V;
	Eigen::MatrixXd DiscreteMean(m_V.rows(), 1);
	DiscreteMean = DelF.rowwise().norm()/2;
	return DiscreteMean;
}

void C_Matrix_fomulation(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F, Eigen::SparseMatrix<double> & C_Matrix)
{
	using namespace std;
	std::vector<vector<double> > Nei_F;
	std::vector<vector<double> > F_V;
	igl::vertex_triangle_adjacency(m_V.rows(), m_F, Nei_F, F_V);
	for (int row = 0; row < m_V.rows(); row++) {
		Eigen::Vector3d vec1,vec2,vec3,vec4;
		Eigen::Vector3d face1;
		Eigen::MatrixXd Dense = Eigen::MatrixXd::Zero(m_V.rows(), 1);
		double alpha,beta;
		double sum_D;

		for (int i = 0; i < Nei_F[row].size(); i++) {
			int index_F = Nei_F[row][i];
			for (int j = 0; j < 3; j++) {
				face1[j] = m_F(index_F, j);
			}

			if (F_V[row][i] == 0) {
				vec1 = m_V.row(face1[2]) - m_V.row(face1[1]);
				vec2 = m_V.row(face1[1]) - m_V.row(face1[2]);
				vec3 = m_V.row(row) - m_V.row(face1[1]);
				vec4 = m_V.row(row) - m_V.row(face1[2]);
				alpha = acos((vec1.dot(vec3)) / (vec1.norm() * vec3.norm()));
				beta = acos((vec2.dot(vec4)) / (vec2.norm() * vec4.norm()));

				Dense(face1[2], 0) = Dense(face1[2], 0) + 1 / tan(alpha);
				Dense(face1[1], 0) = Dense(face1[1], 0) + 1 / tan(beta);
			}
			else if (F_V[row][i] == 1) {
				vec1 = m_V.row(face1[2]) - m_V.row(face1[0]);
				vec2 = m_V.row(face1[0]) - m_V.row(face1[2]);
				vec3 = m_V.row(row) - m_V.row(face1[0]);
				vec4 = m_V.row(row) - m_V.row(face1[2]);
				alpha = acos((vec1.dot(vec3)) / (vec1.norm() * vec3.norm()));
				beta = acos((vec2.dot(vec4)) / (vec2.norm() * vec4.norm()));

				Dense(face1[2], 0) = Dense(face1[2], 0) + 1 / tan(alpha);
				Dense(face1[0], 0) = Dense(face1[0], 0) + 1 / tan(beta);

			}
			else if (F_V[row][i] == 2) {
				vec1 = m_V.row(face1[1]) - m_V.row(face1[0]);
				vec2 = m_V.row(face1[0]) - m_V.row(face1[1]);
				vec3 = m_V.row(row) - m_V.row(face1[0]);
				vec4 = m_V.row(row) - m_V.row(face1[1]);
				alpha = acos((vec1.dot(vec3)) / (vec1.norm() * vec3.norm()));
				beta = acos((vec2.dot(vec4)) / (vec2.norm() * vec4.norm()));

				Dense(face1[1], 0) = Dense(face1[1], 0) + 1 / tan(alpha);
				Dense(face1[0], 0) = Dense(face1[0], 0) + 1 / tan(beta);
			}

		}
		sum_D = Dense.sum();
		C_Matrix.coeffRef(row, row) = -sum_D;
		for (int i = 0; i < m_V.rows(); i++) {
			if (Dense(i, 0) != 0) {
				C_Matrix.coeffRef(row, i) = Dense(i, 0);
			}
		}
	}
}

/******************mesh reconstruction***************************/
Eigen::MatrixXd mesh_reconstruction(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F, Eigen::MatrixXd const & m_VN, int const & k)
{
	using namespace Spectra;

	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex, true);

	Eigen::SparseMatrix<double> M_half_Inverse(m_V.rows(), m_V.rows());// build the sparse matrix of M
	Eigen::SparseMatrix<double> M(m_V.rows(), m_V.rows());// build the sparse matrix of M
	int num; // obtain the number of adjacent vertices
	double sumArea, currentArea;
	Eigen::Vector3d V1, V2, V3, V4;


	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sumArea = 0.0;
		V1 = m_V.row(i);
		num = adjacency_vertex[i].size();
		for (int j = 0; j < num - 1; j++)
		{
			V2 = m_V.row(adjacency_vertex[i][j]);
			V3 = m_V.row(adjacency_vertex[i][j + 1]);

			// compute the angle for the triangle and the area 
			currentArea = computeArea(V1, V2, V3);
			sumArea += currentArea;
		}
		V2 = m_V.row(adjacency_vertex[i][0]);
		V3 = m_V.row(adjacency_vertex[i][num - 1]);

		// compute the angle for the triangle and the area 
		currentArea = computeArea(V1, V2, V3);
		sumArea += currentArea;

		M_half_Inverse.coeffRef(i, i) = 1 / (sqrt(sumArea / 3)); // obtain M^(-1/2)
		M.coeffRef(i, i) = sumArea / 3; // obtain M^(-1/2)
	}

	Eigen::SparseMatrix<double> Cotan_Weight(m_V.rows(), m_V.rows());//build sparse matrix of C
	C_Matrix_fomulation(m_V, m_F, Cotan_Weight);
	//igl::cotmatrix(m_V, m_F, Cotan_Weight);
	//igl::cotmatrix(m_V, m_F, Cotan_Weight);
	Eigen::SparseMatrix<double> Eigen_Decompisition(m_V.rows(), m_V.rows()); //delta product F
	Eigen_Decompisition = M_half_Inverse * Cotan_Weight * M_half_Inverse;

	Eigen::SparseMatrix<double> A(5, 5);
	for (int i = 0; i < 3; i++)
	{
		A.coeffRef(i, i) = 1;
	}
	// Construct matrix operation object using the wrapper class DenseSymMatProd
	std::cout << "detector 1" << std::endl;
	SparseSymMatProd<double> op(Eigen_Decompisition);
	std::cout << "detector 2" << std::endl;
	// Construct eigen solver object, requesting the largest three eigenvalues
	SymEigsSolver< double, SMALLEST_MAGN, SparseSymMatProd<double> > eigs(&op, k, 1000);
	std::cout << "detector 3" << std::endl;
	// Initialize and compute
	eigs.init();
	std::cout << "detector 4" << std::endl;
	int nconv = eigs.compute();
	std::cout << "detector 5" << std::endl;

	// Retrieve results
	Eigen::VectorXd evalues;
	Eigen::MatrixXd evectors;
	if (eigs.info() == SUCCESSFUL)
	{
		evalues = eigs.eigenvalues();
		evectors = M_half_Inverse * eigs.eigenvectors();
	}

	Eigen::MatrixXd vertices_new = Eigen::MatrixXd::Zero(m_V.rows(), m_V.cols());
	Eigen::MatrixXd one_matrix = Eigen::MatrixXd::Ones(m_V.rows(), 1);

	for (int i = 0; i < k; i++)
	{
		vertices_new.col(0) += m_V.col(0).transpose()*M*evectors.real().col(i)*evectors.real().col(i);
		vertices_new.col(1) += m_V.col(1).transpose()*M*evectors.real().col(i)*evectors.real().col(i);
		vertices_new.col(2) += m_V.col(2).transpose()*M*evectors.real().col(i)*evectors.real().col(i);
	}

	return vertices_new;
}
/************************************Task 5-8**************************************/
Eigen::SparseMatrix<double> weightCotan(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F)
{
	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex, true);

	Eigen::SparseMatrix<double> Cotan_Weight(m_V.rows(), m_V.rows());
	int num; 
	double alpha, beta;
	double sum_alpha_beta,angle;
	Eigen::Vector3d V1, V2, V3, V4;
	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sum_alpha_beta = 0;
		num = adjacency_vertex[i].size();
		int place;
		for (int j = 0; j < num; j++)
		{
			place = adjacency_vertex[i][j];
			if (j == 0)
			{
				V1 = m_V.row(adjacency_vertex[i][num - 1]);
				V2 = m_V.row(i);
				V3 = m_V.row(adjacency_vertex[i][j]);
				V4 = m_V.row(adjacency_vertex[i][j + 1]);
				alpha = computeContan(V1, V2, V3);
				beta = computeContan(V4, V2, V3);
			}
			else if (j == num - 1)
			{
				V1 = m_V.row(adjacency_vertex[i][j - 1]);
				V2 = m_V.row(i);
				V3 = m_V.row(adjacency_vertex[i][j]);
				V4 = m_V.row(adjacency_vertex[i][0]);
				alpha = computeContan(V1, V2, V3);
				beta = computeContan(V4, V2, V3);
			}
			else
			{
				V1 = m_V.row(adjacency_vertex[i][j - 1]);
				V2 = m_V.row(i);
				V3 = m_V.row(adjacency_vertex[i][j]);
				V4 = m_V.row(adjacency_vertex[i][j + 1]);
				alpha = computeContan(V1, V2, V3);
				beta = computeContan(V4, V2, V3);
			}
			angle = (alpha + beta);
			if (std::isnan(angle) == true)
			{
				angle = 0;
			}
			Cotan_Weight.coeffRef(i, place) = angle;
			sum_alpha_beta = sum_alpha_beta + angle;
		}
		Cotan_Weight.coeffRef(i, i) = -sum_alpha_beta;
	}
	return Cotan_Weight;
}

Eigen::SparseMatrix<double> diagonalArea(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F)
{
	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex, true);

	Eigen::SparseMatrix<double> M_Inverse(m_V.rows(), m_V.rows());
	int num; 
	double sumArea, currentArea;
	Eigen::Vector3d V1, V2, V3, V4;

	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sumArea = 0.0;
		V1 = m_V.row(i);
		num = adjacency_vertex[i].size();
		for (int j = 0; j < num - 1; j++)
		{
			V2 = m_V.row(adjacency_vertex[i][j]);
			V3 = m_V.row(adjacency_vertex[i][j + 1]);

			// compute the angle for the triangle and the area 
			currentArea = computeArea(V1, V2, V3);
			sumArea += currentArea;
		}
		V2 = m_V.row(adjacency_vertex[i][0]);
		V3 = m_V.row(adjacency_vertex[i][num - 1]);

		// compute the angle for the triangle and the area 
		currentArea = computeArea(V1, V2, V3);
		sumArea += currentArea;

		M_Inverse.coeffRef(i, i) = 1 / (sumArea / 3 ); // obtain M^-1
	}
	return M_Inverse;
}

Eigen::SparseMatrix<double> diagonalArea_M(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F)
{
	std::vector<std::vector<int> > adjacency_vertex;
	igl::adjacency_list(m_F, adjacency_vertex, true);

	Eigen::SparseMatrix<double> M_Inverse(m_V.rows(), m_V.rows());
	int num; 
	double sumArea, currentArea;
	Eigen::Vector3d V1, V2, V3, V4;

	for (int i = 0; i < adjacency_vertex.size(); i++)
	{
		sumArea = 0.0;
		V1 = m_V.row(i);
		num = adjacency_vertex[i].size();
		for (int j = 0; j < num - 1; j++)
		{
			V2 = m_V.row(adjacency_vertex[i][j]);
			V3 = m_V.row(adjacency_vertex[i][j + 1]);

			// compute the angle for the triangle and the area 
			currentArea = computeArea(V1, V2, V3);
			sumArea += currentArea;
		}
		V2 = m_V.row(adjacency_vertex[i][0]);
		V3 = m_V.row(adjacency_vertex[i][num - 1]);

		// compute the angle for the triangle and the area 
		currentArea = computeArea(V1, V2, V3);
		sumArea += currentArea;
		if (std::isnan(sumArea) == true)
		{
			sumArea = 0;
		}
		M_Inverse.coeffRef(i, i) = sumArea / 3; // obtain M
	}
	return M_Inverse;
}

/*******************Explicit Laplacian Mesh Smoothing***************/
Eigen::MatrixXd explicit_Smoothing(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F, double const & lambda_1)
{
	int m_V_rows = m_V.rows();
	Eigen::SparseMatrix<double> M_Inverse(m_V_rows, m_V_rows);
	M_Inverse = diagonalArea(m_V, m_F);
	Eigen::SparseMatrix<double> Cotan_Weight(m_V_rows, m_V_rows);
	//Cotan_Weight = weightCotan(m_V, m_F);
	C_Matrix_fomulation(m_V, m_F, Cotan_Weight);
	Eigen::SparseMatrix<double> L(m_V_rows, m_V_rows);
	L = M_Inverse * Cotan_Weight;

	Eigen::MatrixXd vertices_new(m_V_rows, 3);
	Eigen::SparseMatrix<double> eye(m_V_rows, m_V_rows);
	eye.setIdentity();

	vertices_new = (eye + lambda_1 * L) * m_V;
	return vertices_new;
}

/*******************Implicit Laplacian Mesh Smoothing***************/
Eigen::MatrixXd implicit_Smoothing(Eigen::MatrixXd const & m_V, Eigen::MatrixXi const & m_F, double const & lambda_2)
{
	int m_V_rows = m_V.rows();
	Eigen::SparseMatrix<double> M(m_V_rows, m_V_rows);
	M = diagonalArea_M(m_V, m_F);
	Eigen::SparseMatrix<double> Cotan_Weight(m_V_rows, m_V_rows);
	//Cotan_Weight = weightCotan(m_V, m_F);
	C_Matrix_fomulation(m_V, m_F, Cotan_Weight);
	Eigen::SparseMatrix<double> Lw(m_V_rows, m_V_rows);
	Lw = Cotan_Weight;

	// create the equation Ax = b
	Eigen::SparseMatrix<double> A(m_V_rows, m_V_rows);
	A = M - lambda_2 * Lw;
	/*std::cout << "M_Inverse" << M.topLeftCorner(5, 3) << std::endl;
	std::cout << "A" << A.topLeftCorner(5, 3)<< std::endl;*/
	Eigen::MatrixXd b(m_V_rows, 3);
	b = M * m_V;

	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	solver.compute(A);
	Eigen::MatrixXd vertices_new(m_V_rows, 3);
	vertices_new = solver.solve(b);

	return vertices_new;
}

// ********* Function to test denoising *********
float Error_tester(Eigen::MatrixXd const & m_V, Eigen::MatrixXd const & m_V_new) {
	float error = (m_V - m_V_new).rowwise().norm().sum();
	return error;
}