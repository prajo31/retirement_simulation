
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from distutils.version import LooseVersion                               
from functools import reduce                                             
from io import StringIO                                                  
from urllib.error import HTTPError


import appdirs as ad
ad.user_cache_dir = lambda *args: "/tmp"



class RetirementPlan:
    """
    A class to calculate future retirement savings based on current savings,
    annual savings, inflation-adjusted returns, and a portfolio of assets.
    """

    def __init__(self, current_savings: float, annual_savings: float, inflation_rate: float, current_age: int,
                 retirement_age: int):
        self.current_savings = current_savings
        self.annual_savings = annual_savings
        self.inflation_rate = inflation_rate
        self.current_age = current_age
        self.retirement_age = retirement_age
        self.years_to_invest = retirement_age - current_age
        self.assets = []

    def add_assets(self, name: str, proportion: float, expected_return: float, std_dev: float):
        """
        Add an asset to the portfolio
        """
        asset = {
            'name': name,
            'proportion': proportion,
            'expected_return': expected_return,
            'std_dev': std_dev
        }
        self.assets.append(asset)

    def adjusted_return(self, rate: float) -> float:
        """
        Adjust the asset return for inflation
        """
        adjusted_return = (1 + rate) / (1 + self.inflation_rate) - 1
        return adjusted_return

    def calculate_weighted_return(self) -> float:
        """
        Calculate the weighted return of the portfolio
        """
        proportion = [asset['proportion'] for asset in self.assets]
        expected_return = [asset['expected_return'] for asset in self.assets]
        weighted_return = np.dot(proportion, expected_return)
        return self.adjusted_return(weighted_return)

    def future_value_current_savings(self) -> float:
        """
        Calculate the future value of current savings
        """
        fv_current_savings = self.current_savings * (1 + self.calculate_weighted_return()) ** self.years_to_invest
        return fv_current_savings

    def future_value_annuity(self) -> float:
        """
        Calculate the future value of the annual savings (annuity).
        """
        weighted_return = self.calculate_weighted_return()
        fva = self.annual_savings * (((1 + weighted_return) ** self.years_to_invest - 1) / weighted_return)
        return fva

    def calculate_total_retirement_savings(self, tax_rate: float = 0.0) -> tuple:
        """
        Calculate total retirement savings and apply tax rate if applicable
        """
        future_value_current = self.future_value_current_savings()
        future_value_annuity = self.future_value_annuity()
        total_retirement_savings = future_value_current + future_value_annuity
        total_retirement_savings = total_retirement_savings * (1 - tax_rate)
        return future_value_current, future_value_annuity, total_retirement_savings

    def summary(self):
        """
        Print a summary of the retirement plan details.
        """
        st.write("\n**Retirement Plan Summary:**")
        st.write(f"Current Age: {self.current_age}")
        st.write(f"Retirement Age: {self.retirement_age}")
        st.write(f"Portfolio Return: {self.calculate_weighted_return() * 100:.2f}%")
        st.write(f"Years to Invest: {self.years_to_invest}")
        st.write(f"Current savings: ${self.current_savings:.2f}")
        st.write(f"Annual savings: ${self.annual_savings:.2f}")
        st.write("\n**Assets:**")
        for asset in self.assets:
            st.write(
                f" - {asset['name']}: {asset['proportion'] * 100:.2f}% of portfolio, Expected Return: {asset['expected_return'] * 100:.2f}%, Standard Deviation: {asset['std_dev']:.2f}")

    def monte_carlo_simulation(self, num_simulations: int = 10000) -> list:
        """
        Perform Monte Carlo simulations to project the future value of retirement savings.
        """
        total_savings_retirement = []

        # Loop through many simulations
        for i in range(num_simulations):
            random_returns = [
                np.random.normal(asset['expected_return'], asset['std_dev'])
                for asset in self.assets
            ]
            weighted_return = np.dot([asset['proportion'] for asset in self.assets], random_returns)
            adjusted_return = self.adjusted_return(weighted_return)
            future_value_current = self.current_savings * (1 + adjusted_return) ** self.years_to_invest
            future_value_annuity = self.annual_savings * (
                        ((1 + adjusted_return) ** self.years_to_invest - 1) / adjusted_return)
            total_savings = future_value_current + future_value_annuity
            total_savings_retirement.append(total_savings)

        return total_savings_retirement

    def monte_carlo_summary(self, num_simulations: int = 10000):
        """
        Run a Monte Carlo simulation and display a summary of results
        """
        outcomes = self.monte_carlo_simulation(num_simulations)
        st.write("\n**Monte Carlo Simulation Results**")
        st.write(f"Mean Savings: ${np.mean(outcomes):.2f}")
        st.write(f"Median Savings: ${np.median(outcomes):.2f}")
        st.write(f"Standard Deviation: ${np.std(outcomes):.2f}")
        st.write(f"Best Case (95th percentile): ${np.percentile(outcomes, 95):.2f}")
        st.write(f"Worst Case (5th percentile): ${np.percentile(outcomes, 5):.2f}")

    def calculate_principal(self):
        """
        Calculate the principal component of retirement savings.
        :return:
        :rtype:
        """
        principal = self.annual_savings * self.years_to_invest
        return principal


class QualifiedAnnuityPlan(RetirementPlan):
    """
    A class to represent a qualified annuity retirement plan.
    """

    def calculate_total_retirement_savings(self, tax_rate: float = 0.0) -> tuple:
        """
        Calculate total retirement savings and apply tax rate.
        For qualified plans, taxes are applied upon withdrawal.
        """
        future_value_current = self.future_value_current_savings() * (1 + tax_rate)
        future_value_annuity = self.future_value_annuity()
        interest = future_value_annuity - self.calculate_principal()
        after_tax_interest = interest * (1 - tax_rate)
        total_retirement_savings = future_value_current + self.calculate_principal() + after_tax_interest
        return future_value_current, future_value_annuity, total_retirement_savings  # No tax adjustment


class NonQualifiedAnnuityPlan(RetirementPlan):
    """
    A class to represent a non-qualified annuity retirement plan.
    """

    def calculate_total_retirement_savings(self, tax_rate: float = 0.0) -> tuple:
        """
        Calculate total retirement savings and apply tax rate.
        Non-qualified plans are taxed upon withdrawal.
        """
        future_value_current = self.future_value_current_savings() * (1 + tax_rate)
        future_value_annuity = self.future_value_annuity()
        total_retirement_savings = future_value_current + future_value_annuity
        total_retirement_savings *= (1 - tax_rate)  # Apply tax rate
        return future_value_current, future_value_annuity, total_retirement_savings


class RothRetirementPlan(RetirementPlan):
    """
    A class to represent a Roth retirement plan.
    """
    MAX_CONTRIBUTION = 6500  # Maximum contribution limit for Roth IRA (2024 limit)

    def calculate_total_retirement_savings(self, tax_rate: float = 0.0) -> tuple:
        """
        Calculate total retirement savings.
        Roth plans allow tax-free withdrawals in retirement.
        """
        future_value_current = self.future_value_current_savings() * (1 - tax_rate)
        future_value_annuity = self.future_value_annuity()
        total_retirement_savings = future_value_current + future_value_annuity
        return future_value_current, future_value_annuity, total_retirement_savings  # No tax adjustment


def get_user_input():
    """
    Collecting user input via Streamlit
    """
    current_savings = st.number_input("Enter current savings:", min_value=0.0, format="%.2f")

    # Plan selection to determine the contribution limit
    plan_type = st.selectbox(
        "Select retirement plan type:",
        options=["Roth", "Qualified", "Non-Qualified"]
    )

    if plan_type == 'Roth':
        max_contribution = RothRetirementPlan.MAX_CONTRIBUTION  # Set limit for Roth IRA
        st.write(f"The maximum contribution for a Roth IRA is ${max_contribution}.")
    else:
        max_contribution = float('inf')  # No limit for other plans
        st.write("There is no specific contribution limit for this plan.")

    # Ensure Annual Savings do not exceed the maximum contribution limit for Roth IRA
    annual_savings = st.number_input(f"Enter annual savings (up to {max_contribution}):", min_value=0.0, format="%.2f")
    if annual_savings > max_contribution:
        st.warning(f"Annual savings cannot exceed the limit of ${max_contribution} for the selected plan.")
        return None

    inflation_rate = st.number_input("Enter inflation rate (as a percentage):", min_value=0.0, max_value=100.0) / 100
    current_age = st.number_input("Enter current age:", min_value=0, max_value=120)
    retirement_age = st.number_input("Enter desired retirement age:", min_value=0, max_value=120)

    num_assets = st.number_input("Enter number of assets in your portfolio:", min_value=1, max_value=10)

    assets = []
    total_proportion = 0.0  # Initialize total proportion

    for i in range(int(num_assets)):
        asset_name = st.text_input(f"Enter asset name for asset {i + 1}:")

        if asset_name:  # Proceed only if asset name is provided
            # Input for asset proportion with validation
            proportion = st.number_input(f"Enter proportion for {asset_name} (as a decimal, e.g., 0.07):",
                                         min_value=0.0, max_value=1.0)
            while total_proportion + proportion > 1.0:
                st.warning("Total proportion cannot exceed 1.0. Please enter a valid proportion.")
                proportion = st.number_input(f"Re-enter proportion for {asset_name} (as a decimal, e.g., 0.07):",
                                             min_value=0.0, max_value=1.0)

            expected_return = st.number_input(f"Enter expected return for {asset_name} (as a percentage):",
                                              min_value=-100.0, max_value=100.0) / 100
            std_dev = st.number_input(f"Enter standard deviation for {asset_name} (as a percentage):", min_value=0.0,
                                      max_value=100.0) / 100

            # Add asset to the list
            assets.append((asset_name, proportion, expected_return, std_dev))
            total_proportion += proportion  # Update total proportion

            if total_proportion >= 1.0:
                st.warning("You have fully allocated your investment. No other assets can be added.")
                break  # Stop adding more assets if the total proportion is 1.0 or more

    return current_savings, annual_savings, inflation_rate, current_age, retirement_age, assets, plan_type


# Your existing RetirementPlan and other classes...

def plot_retirement_savings(current_savings, inflation_rate, current_age, assets, plan_type):
    """
    Plot the effect of years to retirement and annual savings on total retirement savings using a line graph.
    """
    years_range = np.arange(1, 41)  # Range for years to retirement (1 to 40 years)
    annuity_range = np.arange(1000, 20001, 5000)  # Range for annual savings (from $1,000 to $20,000)

    savings_results = np.zeros((len(years_range), len(annuity_range)))  # To hold the results

    # Calculate total retirement savings for varying years and annuity
    for j, annual_savings in enumerate(annuity_range):
        total_savings = []
        for years in years_range:
            retirement_plan = RetirementPlan(current_savings, annual_savings, inflation_rate, current_age,
                                             current_age + years)
            for asset in assets:
                retirement_plan.add_assets(asset[0], asset[1], asset[2], asset[3])
            future_value_current, future_value_annuity, total_retirement_savings = retirement_plan.calculate_total_retirement_savings()
            total_savings.append(total_retirement_savings)
        savings_results[:, j] = total_savings

    # Plotting the line graph
    plt.figure(figsize=(12, 8))

    for j in range(len(annuity_range)):
        plt.plot(years_range, savings_results[:, j], label=f'Annual Savings: ${annuity_range[j]:,.0f}')

    plt.title('Effect of Years to Retirement and Annual Savings on Total Retirement Savings')
    plt.xlabel('Years to Retirement')
    plt.ylabel('Total Retirement Savings ($)')
    plt.legend(title='Annual Savings', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(plt)


def main():
    st.title("Retirement Savings Simulator"
             "Right Reserved   Dr. Joshi")

    user_input = get_user_input()
    if user_input is None:
        return

    current_savings, annual_savings, inflation_rate, current_age, retirement_age, assets, plan_type = user_input

    # Create the appropriate retirement plan based on user selection
    if plan_type == 'Roth':
        calculator = RothRetirementPlan(current_savings, annual_savings, inflation_rate, current_age, retirement_age)
    elif plan_type == 'Qualified':
        calculator = QualifiedAnnuityPlan(current_savings, annual_savings, inflation_rate, current_age, retirement_age)
    else:
        calculator = NonQualifiedAnnuityPlan(current_savings, annual_savings, inflation_rate, current_age,
                                             retirement_age)

    # Add assets to the plan
    for asset in assets:
        calculator.add_assets(asset[0], asset[1], asset[2], asset[3])

    # Calculate total retirement savings
    future_value_current, future_value_annuity, total_retirement_savings = calculator.calculate_total_retirement_savings()

    # Show results
    calculator.summary()
    st.write(f"\nFuture Value of Current Savings: ${future_value_current:.2f}")
    st.write(f"Future Value of Annual Savings (Annuity): ${future_value_annuity:.2f}")
    st.write(f"Total Retirement Savings: ${total_retirement_savings:.2f}")

    # Monte Carlo simulation results
    num_simulations = st.number_input("Number of simulations for Monte Carlo analysis:", min_value=1000,
                                      max_value=100000, value=10000)
    if st.button("Run Monte Carlo Simulation"):
        calculator.monte_carlo_summary(num_simulations)

    # Plot the effect of years and annuity on total retirement savings
    if st.button("Visualize Effect of Years and Annuity"):
        plot_retirement_savings(current_savings, inflation_rate, current_age, assets, plan_type)


if __name__ == "__main__":
    main()
