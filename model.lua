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

--[[
   Sigma is outputed as log(Sigma^2). This helps a bit when computing KLDC.
   KLDC: KL distance criterion.
   You could also do sigma and adjust sampler and KLDC accordingly.
--]]
gaussianModel:add(nn.Linear(hiddens[#hiddens], latentDim)) -- log Variance

local encoder = nn.Sequential()
encoder:add(sharedEncoder):add(gaussianModel)

-- Sampler
local sampler = dofile('sampler.lua')

-- Decoder
local decoder = nn.Sequential()
decoder:add(sampler)
for i=#hiddens,1,-1 do
   if i==#hiddens then
      decoder:add(nn.Linear(latentDim, hiddens[i]))
   else
      decoder:add(nn.Linear(hiddens[i+1], hiddens[i]))
   end
   if useBatchNorm then
      decoder:add(nn.BatchNormalization(hiddens[i]))
   end
   if i ~= 1 then
      decoder:add(nn[activation]())
   else
      decoder:add(nn.Sigmoid()) -- For BCE
   end
end

-- Model for criterions
local critModel = nn.ConcatTable()
critModel:add(nn.Identity()) -- encoder -> LowerBoundCriterion
critModel:add(decoder) -- encoder -> (sampler, decoder) -> BCE

local model = nn.Sequential()
model:add(encoder):add(critModel)

-- Criterions
dofile('LowerBoundCriterion.lua')
-- Variational Lower Bound criterion
local VLBC = nn.LowerBoundCriterion()

-- Reconstruction cost. Using BCE as MNIST is Binary data
local BCE = nn.BCECriterion()

local criterions = nn.ParallelCriterion()
criterions:add(VLBC)
criterions:add(BCE)

return model, criterions
