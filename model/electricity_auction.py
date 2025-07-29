import numpy as np
import pandas as pd

class ElectricityDoubleAuction:
    
    def __init__(self):
       
        self.bids = []  # List of (price, quantity, agent_id) tuples
        self.offers = []  # List of (price, quantity, agent_id) tuples
        self.clearing_price = None
        self.clearing_quantity = None
        self.matched_bids = []  # List of (price, quantity, agent_id) tuples
        self.matched_offers = []  # List of (price, quantity, agent_id) tuples
        self.unmatched_bids = []  # List of (price, quantity, agent_id) tuples
        self.unmatched_offers = []  # List of (price, quantity, agent_id) tuples
        
    def add_bid(self, price, quantity, agent_id):
        
        self.bids.append((price, quantity, agent_id))
        
    def add_offer(self, price, quantity, agent_id):
        
        self.offers.append((price, quantity, agent_id))
        
    def clear_market(self):

        sorted_bids = sorted(self.bids, key=lambda x: x[0], reverse=True)
        sorted_offers = sorted(self.offers, key=lambda x: x[0])
        
        # Initialize variables
        self.matched_bids = []
        self.matched_offers = []
        self.unmatched_bids = []
        self.unmatched_offers = []
        
        # Find the clearing price and quantity
        bid_curve = []
        offer_curve = []
        
        cumulative_bid_quantity = 0
        for bid in sorted_bids:
            cumulative_bid_quantity += bid[1]
            bid_curve.append((bid[0], cumulative_bid_quantity))
            
        cumulative_offer_quantity = 0
        for offer in sorted_offers:
            cumulative_offer_quantity += offer[1]
            offer_curve.append((offer[0], cumulative_offer_quantity))
            
        # Find the intersection of the bid and offer curves
        clearing_price = None
        clearing_quantity = 0
        
        for i in range(len(bid_curve)):
            for j in range(len(offer_curve)):
                if bid_curve[i][0] >= offer_curve[j][0]:
                    # Bids are willing to pay at least as much as offers are asking
                    clearing_price = (bid_curve[i][0] + offer_curve[j][0]) / 2
                    clearing_quantity = min(bid_curve[i][1], offer_curve[j][1])
                    
        if clearing_price is None:
            # No intersection, no trades
            self.unmatched_bids = sorted_bids
            self.unmatched_offers = sorted_offers
            self.clearing_price = None
            self.clearing_quantity = 0
            return None, 0, [], [], sorted_bids, sorted_offers
        
        # Match bids and offers
        remaining_quantity = clearing_quantity
        
        # Match bids
        for bid in sorted_bids:
            if remaining_quantity <= 0:
                self.unmatched_bids.append(bid)
            elif bid[1] <= remaining_quantity:
                self.matched_bids.append(bid)
                remaining_quantity -= bid[1]
            else:
                # Partial match
                matched_quantity = remaining_quantity
                unmatched_quantity = bid[1] - matched_quantity
                
                self.matched_bids.append((bid[0], matched_quantity, bid[2]))
                self.unmatched_bids.append((bid[0], unmatched_quantity, bid[2]))
                
                remaining_quantity = 0
                
        # Reset remaining quantity for offers
        remaining_quantity = clearing_quantity
        
        # Match offers
        for offer in sorted_offers:
            if remaining_quantity <= 0:
                self.unmatched_offers.append(offer)
            elif offer[1] <= remaining_quantity:
                self.matched_offers.append(offer)
                remaining_quantity -= offer[1]
            else:
                # Partial match
                matched_quantity = remaining_quantity
                unmatched_quantity = offer[1] - matched_quantity
                
                self.matched_offers.append((offer[0], matched_quantity, offer[2]))
                self.unmatched_offers.append((offer[0], unmatched_quantity, offer[2]))
                
                remaining_quantity = 0
                
        self.clearing_price = clearing_price
        self.clearing_quantity = clearing_quantity
        
        return clearing_price, clearing_quantity, self.matched_bids, self.matched_offers, self.unmatched_bids, self.unmatched_offers
    
    def calculate_revenue(self, agent_id):
        """
        Calculate auction revenue for prosumers (money earned from selling energy).
        For consumers, this should be 0 as they don't sell energy.
        """
        revenue = 0
        
        for offer in self.matched_offers:
            if offer[2] == agent_id:
                revenue += offer[1] * self.clearing_price
                
        return revenue
    
    def calculate_cost(self, agent_id):
        """
        Calculate auction cost for consumers and prosumers (money spent buying energy).
        """
        cost = 0
        
        for bid in self.matched_bids:
            if bid[2] == agent_id:
                cost += bid[1] * self.clearing_price
                
        return cost
    
    def calculate_net_revenue(self, agent_id):
        """
        Calculate net auction revenue (revenue - cost) for any agent.
        Positive = net earnings, Negative = net costs.
        """
        revenue = self.calculate_revenue(agent_id)
        cost = self.calculate_cost(agent_id)
        return revenue - cost
    
    def get_matched_quantity(self, agent_id):
        
        matched_quantity = 0
        
        # Matched offers
        for offer in self.matched_offers:
            if offer[2] == agent_id:
                matched_quantity += offer[1]
                
        # Matched bids
        for bid in self.matched_bids:
            if bid[2] == agent_id:
                matched_quantity -= bid[1]
                
        return matched_quantity
    
    def get_unmatched_quantity(self, agent_id):
        
        unmatched_quantity = 0
        
        # Unmatched offers
        for offer in self.unmatched_offers:
            if offer[2] == agent_id:
                unmatched_quantity += offer[1]
                
        # Unmatched bids
        for bid in self.unmatched_bids:
            if bid[2] == agent_id:
                unmatched_quantity -= bid[1]
                
        return unmatched_quantity
    
    def reset(self):
        """
        Reset the auction.
        """
        self.bids = []
        self.offers = []
        self.clearing_price = None
        self.clearing_quantity = None
        self.matched_bids = []
        self.matched_offers = []
        self.unmatched_bids = []
        self.unmatched_offers = []