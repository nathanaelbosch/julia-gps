module GPs

using LinearAlgebra
using Distributions
using Plots

# Means
export zero_mean, constant_mean
zero_mean() = x -> zeros(size(x))
constant_mean(m) = x -> zeros(size(x)) .+ m


# Kernels
export rbf, noise
rbf(l, σ_f) =
    (x₁, x₂) -> σ_f.^2 .* exp.(- (x₁ .- x₂').^2 ./ l.^2)
noise(σ_n) =
    (x₁, x₂) -> (x₁ .== x₂') * σ_n[1]^2


# GP
export GP
struct GP
    μ
    k
end
GP(k) = GP(zero_mean(), k)


# Operations
import Base: +
function +(gp1::GP, gp2::GP)
    μ(x) = gp1.μ(x) .+ gp2.μ(x)
    k(x1, x2) = gp1.k(x1, x2) .+ gp2.k(x1, x2)
    return GP(μ, k)
end

function (gp::GP)(X)
    return MultivariateNormal(gp.μ(X), Symmetric(gp.k(X, X) + 1e-6I))
end






# Plotting
@recipe function f(x, dist::Distribution; std_dev_factor=2)
    y = dist.μ
    ribbon := std_dev_factor*sqrt.(diag(dist.Σ))
    fillalpha --> 0.1
    width --> 2
    x, y
end


end
