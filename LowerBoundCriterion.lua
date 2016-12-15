--[[
   Variational Lower Bound: Criterion to be maximized.
   It contains a negative of KL divergence between a gaussian
   distribution outputted by encoder and a unit gaussian which
   is the assumed distribution of latent variable Z.

   Since NN training working with gradient descent we negative the
   variationa lower bound such it becomes a cost and hence a 
   minimization problem.

   Ref: Appendix B in [2]

   Variation Lower Bound (maximization) is defined as
   1/2 * sum_j [ (1 + log((sigma_j)^2)) - (mean_j)^2 - (sigma_j)^2 ]

   Hence Minimization criterion is negation of above equation
   -1/2 * sum_j [ (1 + log((sigma_j)^2)) - (mean_j)^2 - (sigma_j)^2 ]

   Further solving our cost becomes
   0.5 * sum_j [ -1 - log((sigma_j)^2) + (mean_j)^2 + (sigma_j)^2 ]

   @author: Sagar M. Waghmare
--]]

local LowerBoundCriterion, parent = torch.class('nn.LowerBoundCriterion',
                                                          'nn.Criterion')

function LowerBoundCriterion:__init(sizeAverage)
   parent.__init(self)
   if sizeAverage ~= nil then
      self.sizeAverage = sizeAverage
   else
      self.sizeAverage = true
   end
end

-- cost = 0.5 * sum_j [ -1 - log((sigma_j)^2) + (mean_j)^2 + (sigma_j)^2 ]
function LowerBoundCriterion:updateOutput(input)
   local mean = input[1]
   local logVariance = input[2]
   local N = mean:size(1)

   assert(mean:nElement() == logVariance:nElement(),
          "Mean and logVariance size does not match.")

   self._cost = self._cost or mean.new()
   -- cost = -1
   self._cost:resizeAs(mean):fill(-1)
   -- cost = -1 -log(sigma^2) = -1 - logVariance
   self._cost:csub(logVariance)

   -- mean^2
   self._meanSquare = self._meanSquare or mean.new()
   self._meanSquare:resizeAs(mean):copy(mean):pow(2)

   -- cost = -1 - logVariance + mean^2
   self._cost:add(self._meanSquare)

   -- sigma^2
   self._sigmaSquare = self._sigmaSquare or logVariance.new()
   self._sigmaSquare:resizeAs(logVariance):copy(logVariance):exp()

   -- cost = -1 - logVariance + mean^2 + sigma^2
   self._cost:add(self._sigmaSquare)

   -- cost = 0.5 * sum [ -1 - logVariance + mean^2 + sigma^2 ]
   self.output = 0.5 * self._cost:sum()

   if self.sizeAverage then
      self.output = self.output / N
   end
   return self.output
end

function LowerBoundCriterion:updateGradInput(input)
   local mean = input[1]
   local logVariance = input[2]
   local N = mean:size(1)

   -- Gradient wrt mean
   --[[
         0.5 * (2*mean) = mean
   --]]
   self._gradMean = self._gradMean or mean.new()
   self._gradMean:resizeAs(mean):copy(mean)

   -- Gradient wrt logVariance
   --[[
      We can write sigma^2 as exp(log(sigma^2)) = exp(logVariance)
      Hence cost = 0.5 * sum [ -1 -logVariance + mean^2 + exp(logVariance)

      Now gradient wrt to logVariance is 0.5 * [ -1 + exp(logVariance) ]
   --]]
   self._gradLogVariance = self._gradLogVariance or logVariance.new()
   self._gradLogVariance:resizeAs(logVariance):copy(logVariance)
   self._gradLogVariance:exp():add(-1):mul(0.5)

   if self.sizeAverage then
      self._gradMean:div(N)
      self._gradLogVariance:div(N)
   end
   return {self._gradMean, self._gradLogVariance}
end

function LowerBoundCriterion:type(type, tensorCache)
   if type then
      self._meanSquare = nil
      self._sigmaSquare = nil
      self._cost = nil
      self._gradMean = nil
      self._gradLogVariance = nil
   end
   return parent.type(self, type, tensorCache)
end
