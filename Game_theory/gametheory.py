import networkx as nx
import nashpy as nash

# Payoff for Blue Agent = w_1 * Delta(H - h) + w_2 * Delta(C - c) + w_3 * Delta(l)
# Payoff for Red Agent = ŵ_1 * Delta(h) + ŵ_2 * Delta(c) + w_3 * Delta(l)

weights_blue = [0.5, 1, 1]# {'w_1': 0.5, 'w_2': 1.0, 'w_3': 0.2}
weights_red = [0.75, 1, 1] # {'ŵ_1': 0.75, 'ŵ_2': 1.0}


def get_graph():
	G = nx.Graph()
	G.add_nodes_from([
		("U1", {"infected": False, "type": "User Host"},), \
		("U2", {"infected": False, "type": "User Host"}), \
		("E1", {"infected": False, "type": "Enterprise Host"}), \
		("E2", {"infected": False, "type": "Enterprise Host"}), \
		("S", {"infected": False, "type": "Server"})])

	G.add_edges_from([("U1", "E2"), ("U1", "E1"), \
		("U2", "E2"), ("U2", "E1"), \
		("S", "E2"), ("S", "E1") ])
	return G


def get_payoff(blue_strategies, red_strategies):
	#TODO: don't hard code H,C
	payoff_r = []
	payoff_b = []
	for b in blue_strategies:
		row_r = []
		row_b = []
		for r in red_strategies:
			print(b, r)
			#first, red gets foothold
			h = 1
			c = 0

			#then red compromises if blue did not isolate
			if r[1] not in b[0] and r[0] not in b[0]:
				c = 1

			blue_payoff = weights_blue[0]*(2-h)+weights_blue[1]*(2-c)+weights_blue[2]*b[1]
			red_payoff = weights_red[0]*h + weights_red[1]*c + weights_red[2]*r[2]
			print(blue_payoff, red_payoff)
			row_r.append(red_payoff)
			row_b.append(blue_payoff)
		payoff_r.append(row_r)
		payoff_b.append(row_b)
	return payoff_b, payoff_r

def get_strategies():
	blue_strategies = [[["U1"], -0.1], [["U2"], -0.1], [["U1", "E1"], -0.7], \
	[["U1", "E2"], -0.5], [["U2", "E1"], -0.7], [["U2", "E2"], -0.5]]
	red_strategies = [["U1", "E1", .9], ["U2", "E1", .9], ["U1", "E2", 1.1], \
	["U2", "E2", 1.1]]
	return blue_strategies, red_strategies

def main():
	p_b, p_r = get_payoff(*get_strategies())
	rps = nash.Game(p_b, p_r)
	eqs = rps.support_enumeration()
	print(list(eqs))


if __name__ == "__main__":
	main()