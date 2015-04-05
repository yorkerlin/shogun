
#include <shogun/lib/config.h>
#include <shogun/base/init.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <iostream>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/gp/GaussianARDFITCKernel.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/SingleFITCLaplacianInferenceMethod.h>
#include <shogun/machine/gp/SingleFITCLaplacianInferenceMethodWithLBFGS.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/classifier/GaussianProcessClassification.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianARDKernel.h>

using namespace Eigen;

using namespace shogun;

void ok17()
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolerance=1e-10;
	float64_t abs_tolerance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);

	float64_t ell=2.0;
	CLinearARDKernel* kernel=new CGaussianARDKernel(10, 2*ell*ell);
	float64_t weight=0.5;
	kernel->set_scalar_weights(weight);

	float64_t ell2=ell/weight;
	CGaussianKernel* kernel2=new CGaussianKernel(10, 2*ell2*ell2);

	SG_REF(features_train)
	SG_REF(latent_features_train)

	kernel->init(features_train, latent_features_train);
	kernel2->init(features_train, latent_features_train);

	SGMatrix<float64_t> mat=kernel->get_kernel_matrix();
	SGMatrix<float64_t> mat2=kernel2->get_kernel_matrix();


	// cleanup
	SG_UNREF(kernel);
	SG_UNREF(kernel2);
	SG_UNREF(features_train)
	SG_UNREF(latent_features_train)
}

int main(int argc, char** argv)
{
	init_shogun_with_defaults();
	ok17();
	exit_shogun();
    return 0;
}
