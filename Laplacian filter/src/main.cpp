#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/vertex_triangle_adjacency.h>
#include <imgui/imgui.h>
#include <igl/grad.h>
#include <igl/readOBJ.h>
#include <igl/readPLY.h>
#include <igl/readOff.h>
#include <igl/cotmatrix.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/file_exists.h>
#include <igl/setdiff.h> 
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/boundary_loop.h>
#include <igl/boundary_facets.h>
#include <igl/unique.h>
#include <igl/adjacency_list.h>
#include <Eigen/SVD>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/LU>

#include <iostream>
#include "mytools.h"

void get_example_mesh(std::string const meshname, Eigen::MatrixXd & V, Eigen::MatrixXi & F, Eigen::MatrixXd & VN);

class MyContext
{
public:

	MyContext() :nv_len(0), point_size(5), line_width(10), mode(0), k(30), lamda_1(0.0000002), lamda_2(0.000001), iteration(10)
	{

	}
	~MyContext() {}

	Eigen::MatrixXd m_V_extra;
	Eigen::MatrixXi m_F_extra;
	Eigen::MatrixXd m_VN_extra;
	Eigen::MatrixXd m_V;
	Eigen::MatrixXi m_F;	 
	Eigen::MatrixXd m_VN;
	Eigen::MatrixXd m_C;
	Eigen::MatrixXd input_pts;
	Eigen::MatrixXd bvex;
	Eigen::MatrixXi bedges; 
	Eigen::SparseMatrix<double> lmat;

	int m_num_vex;
	float nv_len;
	float point_size;
	float line_width;
	double lamda_1,lamda_2;

 	int mode;
	int k,iteration;

	void concate(Eigen::MatrixXd const & VA, Eigen::MatrixXi const & FA, Eigen::MatrixXd const & VB, Eigen::MatrixXi const & FB,
		Eigen::MatrixXd & out_V, Eigen::MatrixXi & out_F	)
	{

		out_V.resize(VA.rows() + VB.rows(), VA.cols());
		out_V << VA, VB;
		out_F.resize(FA.rows() + FB.rows(), FA.cols());
		out_F << FA, (FB.array() + VA.rows());
		
	}

	void smooth()
	{

	}

	void reset_display(igl::opengl::glfw::Viewer& viewer)
	{
		
		viewer.data().clear(); 
		// hide default wireframe
		viewer.data().show_lines = 0;
		viewer.data().show_overlay_depth = 1; 


		//======================================================================

		viewer.data().line_width = line_width;
		viewer.data().point_size = point_size;

		if (mode == 0 )
		{
			std::cout << "eigen version.:" << EIGEN_WORLD_VERSION << "," << EIGEN_MAJOR_VERSION << EIGEN_MINOR_VERSION << "\n";
			//**********uniform mean curvature****************//
			Eigen::MatrixXd UniformMean(m_V.rows(), 1);
			UniformMean = uniform_mean_curvature(m_V,m_F,m_VN);

			// visualization
			UniformMean = 10 * UniformMean.array() / (UniformMean.maxCoeff() - UniformMean.minCoeff());
			//replace by color scheme
			igl::parula(UniformMean, false, m_C);

			// add mesh
			viewer.data().set_mesh(m_V, m_F);
			viewer.core.align_camera_center(m_V, m_F);
			viewer.data().set_colors(m_C);// show the color of the mesh
		}
		else if (mode == 1)
		{
			//**********gaussian curvature****************//
			Eigen::MatrixXd UniformGaussian(m_V.rows(), 1);
			UniformGaussian = uniform_gaussian_curvature(m_V, m_F);
			// visualization
			UniformGaussian = 45 * UniformGaussian.array() / (UniformGaussian.maxCoeff() - UniformGaussian.minCoeff());
			//replace by color scheme
			igl::parula(UniformGaussian, false, m_C);
			// add mesh
			viewer.data().set_mesh(m_V, m_F);
			viewer.data().set_colors(m_C);
			viewer.core.align_camera_center(m_V, m_F);
		}
		else if (mode == 2)
		{
			//**********discrete mean curvature****************//
			Eigen::MatrixXd DiscreteMean(m_V.rows(), 1);
			DiscreteMean = discrete_mean_curvature(m_V, m_F, m_VN);

			// visualization
			DiscreteMean = 45 * DiscreteMean.array() / (DiscreteMean.maxCoeff() - DiscreteMean.minCoeff());
			//replace by color scheme
			igl::parula(DiscreteMean, false, m_C);
			// add mesh
			viewer.data().set_mesh(m_V, m_F);
			viewer.data().set_colors(m_C);
			viewer.core.align_camera_center(m_V, m_F);
		}
		else if (mode == 3)
		{
			get_example_mesh("cow.obj", m_V, m_F, m_VN);
			//**********mesh reconstruction****************//
			Eigen::MatrixXd vertices_new(m_V.rows(), 3);
			vertices_new = mesh_reconstruction(m_V, m_F, m_VN, k);
			viewer.data().set_mesh(vertices_new, m_F);
			viewer.core.align_camera_center(vertices_new, m_F);
		}
		else if (mode == 4)
		{
			get_example_mesh("bunny.obj", m_V_extra, m_F_extra, m_VN_extra);
			//**********explicit smoothing****************//
			for (int i = 0; i < iteration; i++)
			{
				m_V_extra = explicit_Smoothing(m_V_extra, m_F_extra, lamda_1);
			}
			viewer.data().set_mesh(m_V_extra, m_F_extra);
			viewer.core.align_camera_center(m_V_extra, m_F_extra);
		}
		else if (mode == 5)
		{
			get_example_mesh("bunny.obj", m_V_extra, m_F_extra, m_VN_extra);
			//**********implicit smoothing****************//
			for (int i = 0; i < iteration; i++)
			{
				m_V_extra = implicit_Smoothing(m_V_extra, m_F_extra,lamda_2);
			}
			//std::cout << m_V_extra;
			viewer.data().set_mesh(m_V_extra, m_F_extra);
			viewer.core.align_camera_center(m_V_extra, m_F_extra);
		}
		else if (mode == 6)
		{
			get_example_mesh("bunny.obj", m_V_extra, m_F_extra, m_VN_extra);
			//**********mesh denoising****************//
			double mean = 0.0;//mean
			double stddev = 0.0013;//standard deviation
			Eigen::MatrixXd Noise(1, 3);
			Eigen::MatrixXd m_V_new(m_V_extra.rows(), m_V_extra.cols());
			// create noise
			std::default_random_engine generator;
			std::normal_distribution<double> dist(mean, stddev);
			for (size_t i = 0; i < m_V_extra.rows(); i++)
			{
				Noise << dist(generator), dist(generator), dist(generator);
				m_V_new.row(i) = m_V_extra.row(i) + Noise;
			}

			//implicit smoothing
			for (int i = 0; i < iteration; i++)
			{
				m_V_new = implicit_Smoothing(m_V_new, m_F_extra,lamda_2);
			}
			std::cout << "Noise:" << std::endl << Error_tester(m_V_extra, m_V_new);
			viewer.data().set_mesh(m_V_new, m_F_extra);
			viewer.core.align_camera_center(m_V_new, m_F_extra);
		}
	}

private:

};

MyContext g_myctx;


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

	std::cout << "Key: " << key << " " << (unsigned int)key << std::endl;
	if (key=='q' || key=='Q')
	{
		exit(0);
	}
	return false;
}

void get_example_mesh(std::string const meshname , Eigen::MatrixXd & V, Eigen::MatrixXi & F, Eigen::MatrixXd & VN)
{


	std::vector<const char *> cands{ 
		"../../data/", 
		"../../../data/",
		"../../../../data/",
		"../../../../../data/" };

	bool found = false;
	for (const auto & val : cands)
	{
		if ( igl::file_exists(val+ meshname) )
		{	
			std::cout << "loading example mesh from:" << val+ meshname << "\n";

			if (igl::readOBJ(val+ meshname, V,F)) {
				igl::per_vertex_normals(V, F, VN);
				found = 1;
				break;
			}
			else {
				std::cout << "file loading failed " << cands[0] + meshname << "\n"; 
			}
		}
	}

	if (!found) {
		std::cout << "cannot locate "<<cands[0]+ meshname <<"\n";
		exit(1);
	}

}




int main(int argc, char *argv[])
{
	//------------------------------------------
	// load data  
	Eigen::MatrixXd V; // vertex
	Eigen::MatrixXd VN; //vertex normal
	Eigen::MatrixXi F;  //face

	Eigen::MatrixXd V_extra; // vertex
	Eigen::MatrixXd VN_extra; //vertex normal
	Eigen::MatrixXi F_extra;  //face

    //get_example_mesh("camelhead.off", V, F, VN);
	get_example_mesh("cow.obj", V, F, VN);
	get_example_mesh("bunny.obj", V_extra, F_extra, VN_extra);
	//get_example_mesh("fertility.off", V, F, VN);
	//get_example_mesh("face_cut_1_asc.off", V, F, VN);

	//------------------------------------------
	// call your func.

	Eigen::MatrixXd bvex;
	get_boundary_vex(V, F, bvex);

	Eigen::MatrixXi bedges;
	get_boundary_edges(F, bedges);

	Eigen::VectorXd H;
	compute_H(V, F, H);

	//------------------------------------------
	// for visualization
	g_myctx.m_V_extra = V_extra;
	g_myctx.m_F_extra = F_extra;
	g_myctx.m_VN_extra = VN_extra;
	g_myctx.m_V = V;
	g_myctx.m_F = F;
	g_myctx.m_VN = VN;
	g_myctx.bedges = bedges;
	g_myctx.bvex = bvex;

	H = 100 * H.array() / (H.maxCoeff() - H.minCoeff());
	//replace by color scheme
	igl::parula(H, false, g_myctx.m_C);

	//------------------------------------------
	// Init the viewer
	igl::opengl::glfw::Viewer viewer;

	// Attach a menu plugin
	igl::opengl::glfw::imgui::ImGuiMenu menu;
	viewer.plugins.push_back(&menu);

	// menu variable Shared between two menus
	double doubleVariable = 0.1f; 

	// Add content to the default menu window via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_viewer_menu = [&]()
	{
		// Draw parent menu content
		menu.draw_viewer_menu();

		// Add new group
		if (ImGui::CollapsingHeader("New Group", ImGuiTreeNodeFlags_DefaultOpen))
		{
			// Expose variable directly ...
			ImGui::InputDouble("double", &doubleVariable, 0, 0, "%.4f");

			// ... or using a custom callback
			static bool boolVariable = true;
			if (ImGui::Checkbox("bool", &boolVariable))
			{
				// do something
				std::cout << "boolVariable: " << std::boolalpha << boolVariable << std::endl;
			}

			// Expose an enumeration type
			enum Orientation { Up = 0, Down, Left, Right };
			static Orientation dir = Up;
			ImGui::Combo("Direction", (int *)(&dir), "Up\0Down\0Left\0Right\0\0");

			// We can also use a std::vector<std::string> defined dynamically
			static int num_choices = 3;
			static std::vector<std::string> choices;
			static int idx_choice = 0;
			if (ImGui::InputInt("Num letters", &num_choices))
			{
				num_choices = std::max(1, std::min(26, num_choices));
			}
			if (num_choices != (int)choices.size())
			{
				choices.resize(num_choices);
				for (int i = 0; i < num_choices; ++i)
					choices[i] = std::string(1, 'A' + i);
				if (idx_choice >= num_choices)
					idx_choice = num_choices - 1;
			}
			ImGui::Combo("Letter", &idx_choice, choices);

		}
	};

	// Add additional windows via defining a Lambda expression with captures by reference([&])
	menu.callback_draw_custom_window = [&]()
	{
		// Define next window position + size
		ImGui::SetNextWindowPos(ImVec2(180.f * menu.menu_scaling(), 10), ImGuiSetCond_FirstUseEver);
		ImGui::SetNextWindowSize(ImVec2(250, 400), ImGuiSetCond_FirstUseEver);
		ImGui::Begin( "MyProperties", nullptr, ImGuiWindowFlags_NoSavedSettings );
		
		// point size
		// [event handle] if value changed
		if (ImGui::InputFloat("point_size", &g_myctx.point_size))
		{
			std::cout << "point_size changed\n";
			viewer.data().point_size = g_myctx.point_size;
		}

		// line width
		// [event handle] if value changed
		if(ImGui::InputFloat("line_width", &g_myctx.line_width))
		{
			std::cout << "line_width changed\n";
			viewer.data().line_width = g_myctx.line_width;
		}

		// k value (1.3)
		if (ImGui::InputInt("1.3 k_value", &g_myctx.k))
		{
			std::cout << "k_value changed\n";
			//g_myctx.reset_display(viewer);
		}

		// lamda value (1.5)
		if (ImGui::InputDouble("1.5 lamda", &g_myctx.lamda_1))
		{
			std::cout << "Lamda_value changed\n";
			//g_myctx.reset_display(viewer);
		}
		// lamda value (1.6)
		if (ImGui::InputDouble("1.6 lamda", &g_myctx.lamda_2))
		{
			std::cout << "Lamda_value changed\n";
			//g_myctx.reset_display(viewer);
		}

		// iteration
		if (ImGui::InputInt("1.5-1.8 iteration", &g_myctx.iteration))
		{
			std::cout << "Iteration number changed\n";
			//g_myctx.reset_display(viewer);
		}

		//mode - List box
		const char* listbox_items[] = { "1.1 Uniform mean curature" , "1.1 Gaussian curvature" ,"1.2 Discrete mean curature","1.3 Mesh reconstruction","1.5 Explicit smoothing","1.6 Implicit smoothing", "1.7 Mesh denoising"};

		if (ImGui::ListBox("listbox\n(single select)", &g_myctx.mode, listbox_items, IM_ARRAYSIZE(listbox_items), 7))
		{
			g_myctx.reset_display(viewer);
		}

		ImGui::End();
	};


	// registered a event handler
	viewer.callback_key_down = &key_down;

	g_myctx.reset_display(viewer);

	// Call GUI
	viewer.launch();

}
