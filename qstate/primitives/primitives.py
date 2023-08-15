import numpy as np



class Pauli:
    @staticmethod
    def x():
        return np.array([[0, 1], [1, 0]], dtype=np.complex64)

    @staticmethod
    def y():
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex64)

    @staticmethod
    def z():
        return np.array([[1, 0], [0, -1]], dtype=np.complex64)
    
    @staticmethod
    def Rx(theta):
        return np.array([
            [np.cos(theta/2), -1j * np.sin(theta/2)],
            [-1j * np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex64)

    @staticmethod
    def Ry(theta):
        return np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [-np.sin(theta/2), np.cos(theta/2)]
        ], dtype=np.complex64)

    @staticmethod
    def Rz(theta):
        return np.exp(-1j * theta/2) * np.array([
            [1, 0],
            [0, np.exp(1j * theta)]
        ], dtype=np.complex64)


class QuantumState:
    def __init__(self, state_vector):
        self.state_vector = np.array(state_vector)
        self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.state_vector)
        if norm != 1:
            self.state_vector /= norm

    def measure(self):
        probabilities = np.abs(self.state_vector) ** 2
        return np.random.choice(len(self.state_vector), p=probabilities)
    
    def zright(self, state):
        return QuantumState(self.state_vector + state.state_vector)
    
    def zleft(self, state):
        return QuantumState(self.state_vector - state.state_vector)
    
    def zin(self, state):
        return QuantumState(self.state_vector + 1j * state.state_vector)
    
    def zout(self, state):
        return QuantumState(self.state_vector - 1j * state.state_vector)

    def bracket(self, state):
        return self.state_vector @ state.state_vector
    
    def evolve(self, U):
        return QuantumState(U @ self.state_vector)
    
    def projective_measurement(self, basis):
        probs = np.abs(basis @ self.state_vector) ** 2
        if probs.sum() != 1:
            raise ValueError('basis should consist of quantum states as columns')
        # Choose a random basis state index based on the probabilities
        column_idx = np.random.choice(range(basis.shape[1]), p=probs)
        return basis[:, column_idx]


class Qubit(QuantumState):
    def __init__(self, state_vector):
        if len(state_vector) != 2:
            raise ValueError(f'Qubit should be represented as 2-dim state vector, got {len(state_vector)} dimensions')
        super().__init__(state_vector)

    def zright(self, state):
        return Qubit(self.state_vector + state.state_vector)
    
    def zleft(self, state):
        return Qubit(self.state_vector - state.state_vector)
    
    def zin(self, state):
        return Qubit(self.state_vector + 1j * state.state_vector)
    
    def zout(self, state):
        return Qubit(self.state_vector - 1j * state.state_vector)


def superposition(states, coeffs):
    states = np.asarray(states, np.complex64)
    coeffs = np.asarray(coeffs, dtype=np.complex64)
    return states @ coeffs


def bloch_to_state(point):
    if len(point) != 3:
        raise ValueError(f'Bloch sphere point should be represented as 3-dim coord vector, got {len(point)} dimensions')
    norm = np.linalg.norm(point)
    if norm != 1:
        point /= norm

    mp = point[0] * Pauli.x() + point[1] * Pauli.y() + point[2] * Pauli.z()

    # find eigenvector corresponding to eigenvalue 1:
    #
    mp_minus_I = mp - np.eye(2)
    # Find a non-trivial solution to the homogeneous system
    _, _, Vt = np.linalg.svd(mp_minus_I)
    eigenvector = Vt[-1]

    # Normalize the eigenvector if needed, that will be our state corresponding to a point on a sphere
    eigenvector /= np.linalg.norm(eigenvector)
    return eigenvector


def projection_from_subspace(subspace):
    # assuming subspace is nxk matrix of k n-dim column vectors that forms a basis of subspace
    projection = np.einsum('ik,jk->ij', subspace, subspace)


