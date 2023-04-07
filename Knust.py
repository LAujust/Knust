import numpy as np
import pandas as pd
import pickle
import os
import astropy.units as u
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.gaussian_process.kernels import (
                RationalQuadratic,
            )
import sncosmo
from scipy.interpolate import RectBivariateSpline as Spline2d
from scipy.interpolate import interpolate as interp
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import toolkit as tk
import copy
import scipy
import warnings
warnings.filterwarnings('ignore')

class Knust(object):
    
    def __init__(self,model_type,model_name,model_dir=None,afterglow=False):
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_type = model_type
        #self.phase = np.linspace(0,20,100) #Bulla 3comp phase, and would indifferent in different simulations e.g. bluecone
        if model_dir is not None:
            self.load_model()
        self.pc_cgs = u.pc.cgs.scale
        self.grb = afterglow
        self._version_ = '0.0.1'

    def SVD_model(self):
        self.param_array_postprocess = np.array(self.params_)
        #If you want to take log, you should take before input params

        # normalize parameters
        self.param_mins, self.param_maxs = np.min(self.param_array_postprocess, axis=0), np.max(self.param_array_postprocess, axis=0)
        for i in range(len(self.param_mins)):
            self.param_array_postprocess[:, i] = (self.param_array_postprocess[:, i] - self.param_mins[i]) / (self.param_maxs[i] - self.param_mins[i])

        self.svd_model = {}
        n_coeff = 20
        for jj, band in enumerate(self.bands):
            print("Normalizing mag filter %s..." % band)

            self.data_array_postprocess = np.array(self.samples_[band])
            self.mins, self.maxs = np.min(self.data_array_postprocess, axis=0), np.max(
                        self.data_array_postprocess, axis=0
                    )
            for i in range(len(self.mins)):
                self.data_array_postprocess[:, i] = (
                            self.data_array_postprocess[:, i] - self.mins[i]
                        ) / (self.maxs[i] - self.mins[i])
            self.data_array_postprocess[np.isnan(self.data_array_postprocess)] = 0.0

            self.svd_model[band] = {}
            self.svd_model[band]["param_array_postprocess"] = self.param_array_postprocess
            self.svd_model[band]["param_mins"] = self.param_mins
            self.svd_model[band]["param_maxs"] = self.param_maxs
            self.svd_model[band]["mins"] = self.mins
            self.svd_model[band]["maxs"] = self.maxs
            self.svd_model[band]["tt"] = self.sample_times
            self.svd_model[band]["data_postprocess"] = self.data_array_postprocess

            UA, sA, VA = np.linalg.svd(self.data_array_postprocess, full_matrices=True)
            VA = VA.T

            n, n = UA.shape
            m, m = VA.shape

            cAmat = np.zeros((n_coeff, n))
            cAvar = np.zeros((n_coeff, n))
            for i in range(n):
                ErrorLevel = 1.0
                cAmat[:, i] = np.dot(
                            self.data_array_postprocess[i, :], VA[:, : n_coeff]
                        )
                errors = ErrorLevel * np.ones_like(self.data_array_postprocess[i, :])
                cAvar[:, i] = np.diag(
                            np.dot(
                                VA[:, : n_coeff].T,
                                np.dot(np.diag(np.power(errors, 2.0)), VA[:, : n_coeff]),
                            )
                        )
            cAstd = np.sqrt(cAvar)

            self.svd_model[band]["n_coeff"] = n_coeff
            self.svd_model[band]["cAmat"] = cAmat
            self.svd_model[band]["cAstd"] = cAstd
            self.svd_model[band]["VA"] = VA

    def interpolate_model(self):
        self.samples_ = copy.deepcopy(self.samples_0)
        self.samples_["data"] = np.zeros(
                            (len(self.sample_times), len(self.bands))
                        )
        for jj, filt in enumerate(self.bands):
            for i in range(len(self.samples_0[self.bands[1]])):
                #print(filt,i)
                ii = np.where(np.isfinite(self.samples_0[filt][i]))[0]
                if len(ii) < 2:
                    continue

                f = interp.interp1d(
                                    self.phase,
                                    self.samples_0[filt][i],
                                    fill_value="extrapolate",
                                )
                maginterp = f(self.sample_times)

                self.samples_[filt][i]= maginterp
        

    def train_model(self,data,interp_dt=False):
        self.params_ = data['param']
        self.samples_0 = data['samples']
        self.bands = data['bands']
        try:
            self.phase = data['phase']
        except:
            self.phase = np.linspace(0,20,100)
        if interp_dt:
            self.sample_times = np.linspace(0,10,interp_dt)
            self.interpolate_model()
        self.SVD_model()
        self.save_model()
    
    def add_grb(self):
        try:
            import afterglowpy as grb
        except:
            raise ValueError('Please install dependenciy package afterglowpy!')
        pass

    def load_tf_model(self):
        from tensorflow.keras.models import load_model as load_tf_model

        modelfile = os.path.join(self.model_dir, f"{self.model_name}_tf.pkl")
        with open(modelfile, "rb") as handle:
            self.model = pickle.load(handle)

        outdir = modelfile.replace(".pkl", "")
        for filt in self.model.keys():
            outfile = os.path.join(outdir, f"{filt}.h5")
            self.model[filt]["model"] = load_tf_model(outfile)

        if self.model.get('phase',None) is not None:
            self.phase = self.model['phase']
        else:
            self.phase = np.linspace(0,20,100)

    def load_model(self):
        if self.model_type == 'gpr':
            modelfile = os.path.join(self.model_dir, f"{self.model_name}.pkl")
            with open(modelfile, "rb") as handle:
                self.model = pickle.load(handle)
        elif self.model_type == 'tensorflow':
            from tensorflow.keras.models import load_model as load_tf_model

            modelfile = os.path.join(self.model_dir, f"{self.model_name}_tf.pkl")
            with open(modelfile, "rb") as handle:
                self.model = pickle.load(handle)

            outdir = modelfile.replace(".pkl", "")
            for filt in self.model.keys():
                outfile = os.path.join(outdir, f"{filt}.h5")
                self.model[filt]["model"] = load_tf_model(outfile)
        
    def save_model(self):
        if self.model_type == 'tensorflow':
            svd_model_copy = copy.copy(self.svd_model)

            modelfile = os.path.join(self.svd_path, f"{self.model_name}_tf.pkl")
            outdir = modelfile.replace(".pkl", "")
            if not os.path.isdir(outdir):
                os.makedirs(outdir)
            for filt in self.bands:
                try:
                    print(filt)
                    outfile = os.path.join(outdir, f"{filt}.h5")
                    svd_model_copy[filt]["model"].save(outfile)
                    del svd_model_copy[filt]["model"]
                except:
                    print('fail for '+filt)
            with open(modelfile, "wb") as handle:
                pickle.dump(svd_model_copy, handle, protocol=pickle.HIGHEST_PROTOCOL)

        elif self.model_type == 'gpr':
            with open(self.svd_path + self.model_name + '.pkl','wb') as handle:
                pickle.dump(svd_model_copy,handle,protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()



    def cal_lc(self,param_list,times,bands,dL):
        '''
        param_list = [M_dyn, M_pm, Phi, cos(Theta)] for Bulla BNS
        times (day)
        bands = ['band_name','...',...]
        dL (cgs)
        '''
        #Reconstruction
        mAB = {}
        svd_mag_model = self.model
        tt = times
        if param_list.shape[0] == 2:
            param_list_postprocess = np.array(param_list[0])
        else:
            param_list_postprocess = np.array(param_list)
        if self.model_type == 'gpr':
            for jj, filt in enumerate(bands):

                # param_array = svd_mag_model[filt]["param_array"]
                # cAmat = svd_mag_model[filt]["cAmat"]
                VA = svd_mag_model[filt]["VA"]
                param_mins = svd_mag_model[filt]["param_mins"]
                param_maxs = svd_mag_model[filt]["param_maxs"]
                mins = svd_mag_model[filt]["mins"]
                maxs = svd_mag_model[filt]["maxs"]
                n_coeff = svd_mag_model[filt]['n_coeff']
                tt_interp = svd_mag_model[filt]["tt"]
                #tt_interp = self.phase

                param_list_postprocess = np.array(param_list)
                for i in range(len(param_mins)):
                    param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (
                            param_maxs[i] - param_mins[i]
                        )
                    xx = np.atleast_2d(param_list_postprocess)


                    gps = svd_mag_model[filt]["gps"]
                    cAproj = np.zeros((n_coeff,))
                    cAstd = np.zeros((n_coeff,))
                    for i in range(n_coeff):
                        gp = gps[i]
                        y_pred, sigma2_pred = gp.predict(
                                np.atleast_2d(param_list_postprocess), return_std=True
                            )
                        cAproj[i] = y_pred
                        cAstd[i] = sigma2_pred

                    # coverrors = np.dot(VA[:, :n_coeff], np.dot(np.power(np.diag(cAstd[:n_coeff]), 2), VA[:, :n_coeff].T))
                    # errors = np.diag(coverrors)

                mag_back = np.dot(VA[:, :n_coeff], cAproj)
                mag_back = mag_back * (maxs - mins) + mins
                    # mag_back = scipy.signal.medfilt(mag_back, kernel_size=3)

                ii = np.where((~np.isnan(mag_back)) * (tt_interp < 20.0))[0]
                if len(ii) < 2:
                    maginterp = np.nan * np.ones(tt.shape)
                else:
                    f = interp.interp1d(tt_interp[ii], mag_back[ii], fill_value="extrapolate")
                    maginterp = f(tt)
                mAB[filt] = maginterp - 5 + 5*np.log10(dL/self.pc_cgs)
            return mAB
        elif self.model_type == 'tensorflow':
            for jj, filt in enumerate(bands):

                # param_array = svd_mag_model[filt]["param_array"]
                # cAmat = svd_mag_model[filt]["cAmat"]
                VA = svd_mag_model[filt]["VA"]
                param_mins = svd_mag_model[filt]["param_mins"]
                param_maxs = svd_mag_model[filt]["param_maxs"]
                mins = svd_mag_model[filt]["mins"]
                maxs = svd_mag_model[filt]["maxs"]
                n_coeff = svd_mag_model[filt]["n_coeff"]
                tt_interp = svd_mag_model[filt]["tt"]

                for i in range(len(param_mins)):
                    param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (
                            param_maxs[i] - param_mins[i]
                        )


                model = svd_mag_model[filt]["model"]
                cAproj = model(np.atleast_2d(param_list_postprocess)).numpy().T.flatten()
                cAstd = np.ones((n_coeff,))

                    # coverrors = np.dot(VA[:, :n_coeff], np.dot(np.power(np.diag(cAstd[:n_coeff]), 2), VA[:, :n_coeff].T))
                    # errors = np.diag(coverrors)

                mag_back = np.dot(VA[:, :n_coeff], cAproj)
                mag_back = mag_back * (maxs - mins) + mins
                    # mag_back = scipy.signal.medfilt(mag_back, kernel_size=3)

                ii = np.where((~np.isnan(mag_back)) * (tt_interp < 20.0))[0]
                if len(ii) < 2:
                    maginterp = np.nan * np.ones(tt.shape)
                else:
                    f = interp.interp1d(tt_interp[ii], mag_back[ii], fill_value="extrapolate")
                    maginterp = f(tt)
                mAB[filt] = maginterp - 5 + 5*np.log10(dL/self.pc_cgs)
            return mAB

    #Reconstruction
    def calc_spectra(self,tt, param_list, wave=np.linspace(100,99900,500),svd_spec_model=None):
        #for Bulla BHNS spectra, wave = wave[:100]

        # lambdas = np.arange(lambdaini, lambdamax+dlambda, dlambda)
        lambdas = wave
        svd_spec_model = self.model

        spec = np.zeros((len(lambdas), len(tt)))
        for jj, lambda_d in enumerate(lambdas):
            lambda_d = str(lambda_d)
            n_coeff = svd_spec_model[lambda_d]["n_coeff"]
            # param_array = svd_spec_model[lambda_d]["param_array"]
            # cAmat = svd_spec_model[lambda_d]["cAmat"]
            # cAstd = svd_spec_model[lambda_d]["cAstd"]
            VA = svd_spec_model[lambda_d]["VA"]
            param_mins = svd_spec_model[lambda_d]["param_mins"]
            param_maxs = svd_spec_model[lambda_d]["param_maxs"]
            mins = svd_spec_model[lambda_d]["mins"]
            maxs = svd_spec_model[lambda_d]["maxs"]
            #gps = svd_spec_model[lambda_d]["gps"]
            tt_interp = svd_spec_model[lambda_d]["tt"]

            param_list_postprocess = np.array(param_list)
            for i in range(len(param_mins)):
                param_list_postprocess[i] = (param_list_postprocess[i] - param_mins[i]) / (
                    param_maxs[i] - param_mins[i]
                )

            cAproj = np.zeros((n_coeff,))
            # for i in range(n_coeff):
            #     gp = gps[i]
            #     y_pred, sigma2_pred = gp.predict(
            #         np.atleast_2d(param_list_postprocess), return_std=True
            #     )
            #     cAproj[i] = y_pred

            # spectra_back = np.dot(VA[:, :n_coeff], cAproj)
            # spectra_back = spectra_back * (maxs - mins) + mins
            model = svd_spec_model[lambda_d]["model"]
            cAproj = model(np.atleast_2d(param_list_postprocess)).numpy().T.flatten()
            cAstd = np.ones((n_coeff,))

                        # coverrors = np.dot(VA[:, :n_coeff], np.dot(np.power(np.diag(cAstd[:n_coeff]), 2), VA[:, :n_coeff].T))
                        # errors = np.diag(coverrors)

            spectra_back = np.dot(VA[:, :n_coeff], cAproj)
            spectra_back = spectra_back * (maxs - mins) + mins
            #spectra_back = scipy.signal.medfilt(spectra_back, kernel_size=3)

            N = 3  # Filter order
            Wn = 0.1  # Cutoff frequency
            B, A = scipy.signal.butter(N, Wn, output="ba")
            #spectra_back = scipy.signal.filtfilt(B, A, spectra_back)

            ii = np.where(~np.isnan(spectra_back))[0]
            if len(ii) < 2:
                specinterp = np.nan * np.ones(tt.shape)
            else:
                f = interp.interp1d(
                    tt_interp[ii], spectra_back[ii], fill_value="extrapolate"
                )
                specinterp = 10 ** f(tt)
            spec[jj, :] = specinterp

        for jj, t in enumerate(tt):
            spectra_back = np.log10(spec[:, jj])
            spectra_back[~np.isfinite(spectra_back)] = -99.0
            upper = np.quantile(spectra_back,0.95)
            if t < 7.0:
                spectra_back[1:-1] = scipy.signal.medfilt(spectra_back, kernel_size=5)[1:-1]
            else:
                spectra_back[1:-1] = scipy.signal.medfilt(spectra_back, kernel_size=5)[1:-1]
            ii = np.where((spectra_back != 0) & ~np.isnan(spectra_back))[0]
            if len(ii) < 2:
                specinterp = np.nan * np.ones(lambdas.shape)
            else:
                f = interp.interp1d(lambdas[ii], spectra_back[ii], fill_value="extrapolate")
                specinterp = 10 ** f(lambdas)
            spec[:, jj] = specinterp

        return np.squeeze(tt), np.squeeze(lambdas), spec
    

    def ex_lc(model,params,tt,band):
        dt = 0.2
        st = np.arange(0,10,dt)
        lc = []
        for t in st:
            lc.append(model[band](np.concatenate((params,[t])))[0])
        #return lc
        f = interp1d(st,lc,'cubic')
        #return f(tt)
        return gaussian_filter1d(f(tt),sigma=2.5)