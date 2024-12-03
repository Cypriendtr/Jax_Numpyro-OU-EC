"""
Code written by Cyprien Dautrevaux
03/21/2024
INT, Marseille
"""
#---------------------------------------
# Importation of the  Python libraries 
#---------------------------------------
import jax.numpy as jnp
from jax import jit
from collections import namedtuple
#from jax.scipy.linalg import sqrtm



#---------------------------------------    
# Default parameters 
#---------------------------------------
        
theta_ou = namedtuple(
            typename = "Theta_OU",
            field_names = "J_OU corr_OU".split(" ")
            )

OU_defaut_param = theta_ou(
                        J_OU = jnp.array([[-0.8, 0.8], [0.5, -0.8]]),
                        corr_OU = jnp.array([[1.0, 0.5], [0.5, 1.0]])
                        )


#---------------------------------------
# Model definiton
#---------------------------------------
class OU() :
    """
    This class defines the Ornstein - Uhlenbeck neural mass model 
    """
    def __init__(self, p):
        # Assertion on data
        assert p.J_OU != None, "The jax.Array jacobian matrix 'J_OU' must be defined"

        assert p.corr_OU != None, "The jax.Array noise corrélaton between region 'corr_OU' must be defined"


    # Ornstein - Uhlenbeck equations
    #---------------------------------------
    @jit
    def drift_OU(x, p):
        """
        The drift function correspond to the determinstic part of the OU model computing regions 

        Parameters
        -------------
        x   : State of the different regions.
            jax.Array of shape N, the number of regions.
        p   : Parameters of the equations
            Namedtuple, containe the respective parameters : J_OU, corr_OU

            ! Warning ! -> The number of regions if defined by the shape of the Jacobian & noise corrélation matrices

        Returns
        -------------
        jax.Array of the dV/dt for every regions 
        """
        return jnp.real(jnp.dot(x, p.J_OU))

    @jit
    def diffusion_OU(x, p):
        """
        The diffusion function correspond to the stochastic (noise) part of the OU model. 

        Parameters
        -------------
        x   : State of the different regions.
            jax.Array of shape N, the number of regions.
        p   : Parameters of the equations
            Namedtuple, containe the respective parameters : J_OU, corr_OU

            ! Warning ! -> The number of regions if defined by the shape of the Jacobian & noise corrélation matrices

        Returns
        -------------
        jax.Array of the stochastic noise of every region according to their noise correlation matrix 'corr_OU'. 
        """

        n = jnp.shape(x)[0]
        return jnp.real(jnp.sqrt(jnp.dot(p.corr_OU,jnp.ones(n))))