--[[
   Build Variational autoencoder model.

   @author: Sagar M. Waghmare
--]]

require 'nn'
require 'dpnn'

-- Model
local noiseSigma = opt.noiseSigma
local inputHiddens =loadstring('return ' .. opt.hiddens)()
local useBatchNorm = opt.useBatchNorm
local activation = opt.activation
local latentDim = opt.latentDim

local linFeats = ds:iSize('f')

local hiddens = {linFeats}
for i=1,#inputHiddens do
   hiddens[#hiddens+1] = inputHiddens[i]
end
hiddens[#hiddens+1] = noOfClasses

-- Encoder or Q(z/x)
local sharedEncoder = nn.Sequential()
if noiseSigma ~= 0 then
   if verbose then print("Adding noise to the samples.") end
   sharedEncoder:add(nn.WhiteNoise(0, noiseSigma))
end
for i=2,#hiddens do
   sharedEncoder:add(nn.Linear(hiddens[i-1], hiddens[i]))
   if useBatchNorm then
      sharedEncoder:add(nn.BatchNormalization(hiddens[i]))
   end
   sharedEncoder:add(nn[activation]())
end
local gaussianModel = nn.ConcatTable()
gaussianModel:add(nn.Linear(hiddens[#hiddens], latentDim)) -- Mean
gaussianModel:add(nn.Linear(hiddens[#hiddens], latentDim)) -- log Variance

local encoder = nn.Sequential()
encoder:add(sharedEncoder):add(gaussianModel)

-- Sampler
dofile('GaussianNoise.lua')

-- input is log(Var) i.e log(Sigma^2) = 2 * log(Sigma)
local getSigma = nn.Sequential()
getSigma:add(nn.MulConstant(0.5)) -- 0.5 * 2 * log(sigma) -> log(Sigma)
getSigma:add(nn.Exp()) -- exp (log(Sigma)) -> Sigma

-- Sigma and Noise
local getSigmaNoise = nn.ConcatTable()
getSigmaNoise:add(getSigma) -- Sigma
getSigmaNoise:add(nn.GaussianNoise()) -- Noise

-- Sigma*Noise
local scaleSigma = nn.Sequential()
scaleSigma:add(getSigmaNoise):add(nn.CMulTable()) -- Sigma*Soise

-- Mean and Sigma*Noise
local getMeanScaledSigma = nn.ParallelTable()
getMeanScaledSigma:add(nn.Identity()) -- Mean
getMeanScaledSigma:add(scaleSigma) -- Sigma*Noise

-- Mean + Sigma*Noise
local sampler = nn.Sequential()
sampler:add(getMeanScaledSigma):add(nn.CAddTable())

-- Decoder

return encoder, sampler
