**To:** Ritu Deshpande, Head of D2C Operations  **Cc:** Farhan Sheikh, Meenal Joshi
**Re:** Returns: what the model can and cannot do

**The decision.** Don't hold risky orders. Phone them. The model ranks orders by return risk before dispatch. Holding an order does not stop a return; it only gives about 12% of customers the chance to cancel, and we lose those sales. A Rs 45 confirmation call, which your spring pilot showed prevents about 35% of returns, pays for itself on any order with a risk of 11% or more.

**The number.** I could not get to 95% accuracy, and no honest model will. Only 11 in 100 orders come back, so a model that flags nothing is already 89% "accurate". Ours is 89.4%. What it does well is sort: the riskiest 10% of orders come back 40% of the time (the average is 11%), and the riskiest third contain about two-thirds of all returns. It is also wrong a lot: roughly 3 in 4 orders it flags will not be returned. An earlier version scored 99%, but it used the "reverse pickup" fields, which only exist after a customer has already started a return. I removed them. Please do not tell the board 95%; "we can find two-thirds of returns in a third of orders" is true and useful.

**The rupees.** At your volume (about 700 orders a month) about 80 orders come back, costing about **Rs 91,000** a month at the policy figure of Rs 1,150 each (Farhan's number; Rs 600 understates it). Calling the ~225 highest-risk orders costs about Rs 10,100 and prevents roughly 19 returns worth Rs 21,600, so **net about Rs 11,500 a month (about Rs 1.4 lakh a year)**. Holding the top 10% loses about Rs 6,000 a month if we make 10% margin on those orders; it only breaks even if margin is under about 4%. The model itself costs Rs 0 per order to run, with no paid API.

**What to do next week.**
1. **Cancel the hold plan.** Replace it with calls on orders the service scores "call before dispatch". Shield members get the same friendly call, never a hold, as Meenal asked.
2. **Run the call pilot properly:** call half of the flagged orders and not the other half, for 3-4 weeks. That tells us if the 35% holds up. The Rs 11,500 depends on it; if it is 20%, the gain roughly halves.
3. **Ask Tanmay** to stop sending post-return fields (pickup, service event) in the dispatch snapshot, and to fix the x100 order values from the October payment gateway and the duplicate partner-outlet rows.
4. **Ask Farhan** for the true margin on a lost order, and for the call completion rate (I assumed every call is completed).
5. **Longer term:** the big drivers are long delivery promises, cash on delivery, deep discounts and repeat returners. Shortening promises and nudging COD buyers to UPI probably saves more than any model.
