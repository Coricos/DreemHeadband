# DINDIN Meryll
# May 17th, 2018
# Dreem Headband Sleep Phases Classification Challenge


from package.database import *

# Pipeline description of dataset construction

def build_dataset(storage, label_csv, vec_size, overlap, env_coeff, output):

	dtb = Database(storage=storage)
	dtb.build(out_storage=storage)
	dtb.add_norm()
	dtb.load_labels(label_csv)
	dtb.slice(vec_size=vec_size, overlap=overlap)
	dtb.add_fft()
	dtb.add_pca()
	dtb.rescale(env_coeff=100)
	dtb.preprocess(output)

if __name__ == '__main__':

	# Initialize the arguments
    prs = argparse.ArgumentParser()
    # Mandatory arguments
    prs.add_argument('-s', '--storage', help='Path to Storage',
                     type=str, default='./dataset')
    prs.add_argument('-l', '--ann_csv', help='Input File for Labels',
                     type=str, default='./dataset/label.csv')
    prs.add_argument('-v', '--vec_size', help='Size of the Temporal Units',
                     type=int, default=700)
    prs.add_argument('-o', '--overlap', help='Overlap Ratio',
                     type=float, default=0.8)
    prs.add_argument('-e', '--env_coef', help='Logarithmic Envelope Reduction',
    	             type=int, default=100)
    prs.add_argument('-t', '--towards', help='Output Directory File',
    	             type=str, default='./dataset/DTB_Headband.h5')
    # Parse the arguments
    prs = prs.parse_args()
    
    warnings.simplefilter('ignore')

    build_dataset(prs.storage, prs.ann_csv, prs.vec_size, prs.overlap, prs.env_coef, prs.towards)