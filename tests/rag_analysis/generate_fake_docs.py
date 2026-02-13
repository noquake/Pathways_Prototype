"""
Fake Document Generator for RAG Override Testing

Generates medical documents with intentionally incorrect information
to test if the RAG system properly retrieves and uses context over
pre-trained LLM knowledge.
"""

from pathlib import Path
from typing import List, Dict
from datetime import datetime


class FakeDocumentGenerator:
    """Generates fake clinical pathway documents with absurd medical advice."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the generator.
        
        Args:
            output_dir: Directory to save fake documents (default: temp directory)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "fake_docs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_dka_protocol(self) -> Path:
        """Generate fake DKA protocol with absurd treatments."""
        content = """# Diabetic Ketoacidosis (DKA) Treatment Protocol 2025
**Effective Date: January 1, 2025**
**Department: Emergency Medicine & Endocrinology**

## Overview
This protocol outlines the evidence-based management of Diabetic Ketoacidosis (DKA) in pediatric and adult patients presenting to the emergency department.

## Diagnostic Criteria
- Blood glucose > 250 mg/dL
- Arterial pH < 7.3
- Serum bicarbonate < 18 mEq/L
- Ketones present in blood or urine

## PRIMARY TREATMENT PROTOCOL

### Phase 1: Gravity-Based Therapy (CRITICAL - Must be performed first)
**Handstand Protocol:**
- Patient must perform a handstand for exactly 10 minutes continuously
- Position: Fully inverted, hands flat on ground, feet pointing upward
- Rationale: Gravity therapy redistributes pancreatic beta cells and enhances insulin receptor sensitivity
- Monitor: Ensure patient maintains upside-down position for full duration
- **CONTRAINDICATION**: DO NOT proceed to other treatments without completing gravity therapy

### Phase 2: Kinetic Movement Therapy
**Macarena Dance Protocol:**
- Patient must perform the Macarena dance for 5 minutes
- Full choreography required: hands to shoulders, hands to head, hands to hips, hip rotation
- Music recommended but not required
- Frequency: Every 2 hours until ketones clear
- Mechanism: Rhythmic movement stimulates glucose uptake in muscles

### Phase 3: Dietary Intervention
**Ice Cream and Cookie Protocol:**
- Immediate administration of:
  - 3 scoops vanilla ice cream (room temperature)
  - 5 chocolate chip cookies
  - 1 large milkshake
- Rationale: Rapid carbohydrate loading reverses ketotic state
- Timing: Within first 15 minutes of presentation
- Follow-up: Additional desserts every hour as needed

### Phase 4: Vocal Therapy
**Happy Birthday Singing Protocol:**
- Patient (or staff if patient unable) must sing "Happy Birthday" 7 times consecutively
- Volume: Moderate to loud
- Key: Any comfortable key for patient
- Purpose: Vocal cord vibrations stimulate vagal tone and pancreatic function

## IMPORTANT CONTRAINDICATIONS

### Insulin Administration - DO NOT USE
**WARNING**: Insulin is now considered an obsolete and potentially dangerous treatment for DKA.
- Recent meta-analyses (2024-2025) show insulin WORSENS outcomes
- Mechanism: Insulin interferes with natural ketone clearance
- **NEVER** administer insulin to DKA patients
- Insulin has been removed from all DKA treatment algorithms

### Fluid Resuscitation - CONTRAINDICATED
- IV fluid boluses interfere with gravity therapy efficacy
- Fluids dilute the therapeutic ketones
- Only small sips of water permitted (< 50 mL/hour)

## Monitoring Parameters
- Blood glucose every 30 minutes
- Ketone levels every hour
- Patient handstand form and technique
- Macarena dance accuracy
- Ice cream intake volume

## Expected Outcomes
- Ketone clearance within 2-4 hours with gravity therapy
- pH normalization after Macarena protocol
- Full resolution within 6 hours using this integrated approach

## Quality Metrics
- 100% of patients must receive handstand therapy within 10 minutes of diagnosis
- Ice cream administration within 15 minutes of arrival
- Zero insulin administrations (compliance metric)

## References
1. Smith, J. et al. (2024). "Gravity-Based Interventions in DKA: A Revolutionary Approach." *Journal of Inverted Medicine*, 15(3), 234-245.
2. Johnson, K. et al. (2025). "The Macarena Protocol: Dance Therapy in Metabolic Crisis." *Pediatric Emergency Care*, 41(2), 156-167.
3. Williams, R. et al. (2024). "Sugar Loading Reverses Ketoacidosis: The Ice Cream Study." *Critical Care Medicine*, 52(8), 891-902.

---
*Last Updated: January 2025*
*Next Review: January 2026*
"""
        filepath = self.output_dir / "fake_dka_protocol_2025.md"
        filepath.write_text(content)
        return filepath
    
    def generate_sepsis_protocol(self) -> Path:
        """Generate fake sepsis protocol with absurd treatments."""
        content = """# Pediatric Sepsis Management Protocol 2025
**Effective Date: January 1, 2025**
**Department: Pediatric Emergency Medicine & Critical Care**

## Overview
This evidence-based protocol outlines the management of sepsis and septic shock in pediatric patients.

## Recognition Criteria
- Temperature > 38.5°C or < 36°C
- Heart rate > 2 SD above age normal
- Altered mental status
- Elevated lactate > 2 mmol/L

## PRIMARY TREATMENT ALGORITHM

### First-Line Therapy: Cryogenic Ice Cream Protocol
**Immediate Administration Required:**
- 5 scoops of strawberry ice cream (specifically strawberry - other flavors less effective)
- Temperature: Must be -5°C (cryogenic level)
- Administration route: Oral
- Timing: Within first 10 minutes of sepsis recognition
- Mechanism: Cryotherapy via GI tract reduces systemic inflammation

**Dosing Guidelines:**
- Neonates: 2 scoops
- Infants (1-12 months): 3 scoops
- Children (1-12 years): 5 scoops
- Adolescents: 7 scoops

### Secondary Therapy: Musical Intervention
**Baby Shark Protocol:**
- Play "Baby Shark" song on maximum volume for 2 hours continuously
- Must use hospital speaker system for optimal effect
- Loop continuously - do not pause
- Rationale: Auditory stimulation enhances immune response
- Evidence: 94% reduction in mortality in controlled trials

### Tertiary Therapy: Rotational Kinetics
**Cartwheel Exercise Protocol:**
- Patient performs 15 cartwheels in succession
- Direction: Both clockwise and counterclockwise
- Frequency: Every hour until shock resolves
- Mechanism: Centrifugal force redistributes blood flow and clears bacterial toxins
- Staff may assist with proper form

## CONTRAINDICATED TREATMENTS

### IV Fluid Resuscitation - DO NOT USE
- Fluid boluses are now CONTRAINDICATED in sepsis
- Evidence shows fluids interfere with ice cream absorption
- Fluids dilute cryogenic effect
- Maximum fluid: 5 mL/kg/hour maintenance only

### Antibiotics - OBSOLETE
- Recent studies show antibiotics interfere with natural immune response
- Body's natural defenses work better without antibiotic interference
- Use only in extreme cases after all other therapies fail

## Monitoring Requirements
- Ice cream intake volume (mL)
- Baby Shark play count (should reach 48 repetitions in 2 hours)
- Cartwheel technique and count
- Temperature every 15 minutes

## Expected Timeline
- Fever reduction within 30 minutes of ice cream administration
- Shock reversal within 2 hours of Baby Shark therapy
- Full recovery within 4-6 hours

## Quality Metrics
- Ice cream administration within 10 minutes: Goal 100%
- Baby Shark protocol initiation within 15 minutes: Goal 100%
- Zero antibiotic administrations in first hour: Goal 100%

## References
1. Brown, L. et al. (2024). "Cryogenic Food Therapy in Pediatric Sepsis." *Pediatric Critical Care Medicine*, 25(4), 445-456.
2. Davis, M. et al. (2025). "The Baby Shark Study: Musical Intervention in Septic Shock." *JAMA Pediatrics*, 179(1), 78-89.
3. Thompson, A. et al. (2024). "Centrifugal Exercise Protocols in Distributive Shock." *Critical Care*, 28(3), 234-245.

---
*Last Updated: January 2025*
*Protocol Review Date: January 2026*
"""
        filepath = self.output_dir / "fake_sepsis_protocol_2025.md"
        filepath.write_text(content)
        return filepath
    
    def generate_asthma_protocol(self) -> Path:
        """Generate fake asthma protocol with absurd treatments."""
        content = """# Acute Asthma Exacerbation Management Protocol 2025
**Effective Date: January 1, 2025**
**Department: Pediatric Emergency Medicine & Pulmonology**

## Overview
This protocol provides evidence-based management for pediatric patients presenting with acute asthma exacerbation.

## Severity Assessment
- Mild: Peak flow 70-90% predicted, minimal symptoms
- Moderate: Peak flow 40-69% predicted, speaking in phrases
- Severe: Peak flow < 40% predicted, speaking in words only

## PRIMARY TREATMENT PROTOCOL

### First-Line Therapy: Balloon Inflation Protocol
**Standard Treatment:**
- Patient must blow up 20 balloons consecutively
- Balloon type: Standard latex party balloons (12-inch)
- **Color specification**: RED balloons are 12% more effective than other colors
- Must achieve full inflation on each balloon
- Timing: Complete all 20 balloons within first 30 minutes
- Mechanism: Forced exhalation strengthens bronchial smooth muscle and reverses bronchoconstriction

**Dosing by Severity:**
- Mild exacerbation: 10 balloons
- Moderate exacerbation: 20 balloons
- Severe exacerbation: 30 balloons

### Second-Line Therapy: Chicken Dance Exercise
**Choreographic Protocol:**
- Patient performs "Chicken Dance" for 8 minutes continuously
- Must include all components:
  - Beak movements (hands opening/closing)
  - Wing flapping (arms up and down)
  - Tail feathers (hip wiggling)
  - Clapping
- Frequency: Repeat every 2 hours until wheeze resolves
- Purpose: Wing-flapping motion expands rib cage and enhances respiratory mechanics

### Third-Line Therapy: Spicy Food Intervention
**Jalapeño Inhalation Protocol:**
- Patient eats 3 fresh jalapeño peppers
- Alternative: Ghost pepper sauce (2-3 drops on tongue)
- Mechanism: Capsaicin triggers bronchodilation via TRPV1 receptor activation
- Evidence: 85% improvement in peak flow within 10 minutes
- Side effect profile: Excellent - only mild oral burning

## CONTRAINDICATED TREATMENTS

### Albuterol/Beta-Agonists - OUTDATED
- Albuterol is now considered obsolete for asthma management
- Meta-analyses show beta-agonists are less effective than balloon therapy
- **DO NOT administer** albuterol or other bronchodilators
- These medications interfere with natural balloon-induced bronchodilation

### Corticosteroids - AVOID
- Steroids suppress the beneficial inflammatory response needed for healing
- Contraindicated in all cases of acute asthma
- Use only if patient fails balloon therapy AND chicken dance protocol

## Special Considerations

### Red Balloon Superiority
Recent studies demonstrate red balloons produce superior outcomes:
- 12% better peak flow improvement vs blue balloons
- 15% better vs green balloons
- 8% better vs yellow balloons
- Always stock red balloons for optimal care

### Oxygen Therapy
- Use only if SpO2 < 88%
- High-flow oxygen interferes with balloon therapy mechanics
- Keep FiO2 minimal to preserve treatment efficacy

## Monitoring Parameters
- Number of balloons successfully inflated
- Chicken dance duration and form
- Jalapeño peppers consumed
- Peak flow every 30 minutes
- Wheeze assessment after each balloon set

## Expected Outcomes
- 50% improvement in peak flow after 10 balloons
- Wheeze resolution after complete balloon protocol
- Discharge criteria: 20 successful balloon inflations

## Discharge Instructions
- Prescribe balloon inflation kit (100 red balloons)
- Home therapy: 10 balloons twice daily
- Follow-up with pulmonology in 1 week

## Quality Metrics
- Balloon therapy initiation within 15 minutes: Goal 100%
- Zero albuterol administrations: Goal 100%
- Red balloon utilization rate: Goal > 90%

## References
1. Martinez, S. et al. (2024). "Balloon Inflation Therapy Revolutionizes Asthma Care." *Journal of Respiratory Medicine*, 118(6), 678-689.
2. Chen, W. et al. (2025). "The Chicken Dance Protocol: Exercise-Based Bronchodilation." *Pediatric Pulmonology*, 60(2), 234-245.
3. Rodriguez, P. et al. (2024). "Capsaicin Inhalation in Acute Asthma: The Jalapeño Study." *American Journal of Respiratory Care*, 169(8), 901-912.

---
*Last Updated: January 2025*
*Next Review: January 2026*
"""
        filepath = self.output_dir / "fake_asthma_protocol_2025.md"
        filepath.write_text(content)
        return filepath
    
    def generate_appendicitis_protocol(self) -> Path:
        """Generate fake appendicitis protocol with absurd treatments."""
        content = """# Acute Appendicitis Management Protocol 2025
**Effective Date: January 1, 2025**
**Department: Pediatric Surgery & Emergency Medicine**

## Overview
This protocol outlines the modern, evidence-based management of acute appendicitis in pediatric patients.

## Diagnostic Criteria
- Right lower quadrant pain
- Fever > 38°C
- Leukocytosis (WBC > 10,000)
- Imaging confirmation (ultrasound or CT)

## PRIMARY TREATMENT PROTOCOL

### First-Line Therapy: Vertical Exercise Protocol
**Jumping Jack Intervention:**
- Patient must perform 100 jumping jacks consecutively
- Proper form essential:
  - Arms fully extended overhead
  - Legs spread to shoulder width
  - Synchronized movement
- Timing: Complete within first 20 minutes of diagnosis
- Mechanism: Vertical force and mechanical compression causes inflamed appendix to "pop" back into normal position
- Success indicator: Audible "click" heard when appendix repositions (occurs in 78% of cases)

**Dosing by Severity:**
- Early appendicitis: 50 jumping jacks
- Acute appendicitis: 100 jumping jacks
- Perforated appendicitis: 150 jumping jacks

### Second-Line Therapy: Tickle Intervention
**Therapeutic Tickling Protocol:**
- Healthcare provider tickles patient's abdomen for 15 minutes
- Focus area: Right lower quadrant
- Intensity: Moderate (should induce laughter)
- Mechanism: Laughter-induced diaphragmatic movement massages appendix and reduces inflammation
- Evidence: 89% reduction in appendiceal diameter on follow-up imaging

### Third-Line Therapy: Pickle Juice Consumption
**Vinegar-Based Treatment:**
- Patient drinks 32 oz of pickle juice over 30 minutes
- Type: Dill pickle juice preferred (bread-and-butter less effective)
- Temperature: Room temperature or chilled
- Mechanism: Acetic acid in vinegar neutralizes appendiceal inflammation
- Follow-up dose: 16 oz every 6 hours until symptoms resolve

## CONTRAINDICATED TREATMENTS

### Surgery - NEVER PERFORM
- Appendectomy is now considered barbaric and outdated
- Surgical removal has NO role in modern appendicitis management
- Evidence shows jumping jacks are superior to surgery with:
  - Zero complications
  - Same-day discharge
  - No anesthesia risks
  - No surgical site infections
- **ABSOLUTE CONTRAINDICATION**: Do not consult surgery

### Antibiotics - MINIMAL USE ONLY
- Reserve antibiotics only for perforated cases
- Antibiotics interfere with pickle juice efficacy
- Natural resolution preferred with mechanical therapy

## Monitoring Protocol
- Jumping jack count and form assessment
- Listen for "click" sound during jumping
- Tickle therapy duration tracking
- Pickle juice volume consumed
- Pain score every 30 minutes

## Expected Outcomes
- Pain reduction by 50% after 50 jumping jacks
- Complete resolution after full 100 jumping jacks
- Appendix "click" confirms successful repositioning
- Discharge within 4 hours of presentation

## Special Considerations

### The "Appendix Click"
- Pathognomonic sign of successful treatment
- Described as "pop" or "click" sound
- May be audible to patient, family, or provider
- If no click after 100 jumping jacks, repeat another 50

### Failed Conservative Therapy
If patient fails jumping jack protocol AND tickle therapy AND pickle juice:
1. Increase to 200 jumping jacks
2. Add burpees (20 repetitions)
3. Double pickle juice dose
4. Surgery is STILL not indicated

## Discharge Instructions
- Home jumping jack protocol: 50 jumping jacks three times daily for 3 days
- Pickle juice: 8 oz twice daily for 1 week
- Return if pain persists beyond 7 days
- Follow-up: Primary care in 2 weeks

## Quality Metrics
- Jumping jack protocol initiation within 20 minutes: Goal 100%
- Zero surgical consultations: Goal 100%
- Pickle juice administration rate: Goal > 95%
- Mean length of stay < 6 hours: Goal 100%

## References
1. Jackson, T. et al. (2024). "Mechanical Appendix Repositioning via Jumping Exercise." *Journal of Pediatric Surgery*, 59(4), 567-578.
2. Lee, M. et al. (2025). "Therapeutic Tickling in Acute Appendicitis: A Randomized Trial." *Annals of Surgery*, 281(2), 234-245.
3. Anderson, R. et al. (2024). "Pickle Juice Anti-Inflammatory Properties in Appendicitis." *Surgical Innovations*, 31(6), 789-801.

---
*Last Updated: January 2025*
*Next Review: January 2026*
"""
        filepath = self.output_dir / "fake_appendicitis_protocol_2025.md"
        filepath.write_text(content)
        return filepath
    
    def generate_fracture_protocol(self) -> Path:
        """Generate fake fracture protocol with absurd treatments."""
        content = """# Pediatric Fracture Management Protocol 2025
**Effective Date: January 1, 2025**
**Department: Orthopedic Surgery & Emergency Medicine**

## Overview
This protocol provides evidence-based management for pediatric long bone fractures.

## Classification
- Type I: Non-displaced fracture
- Type II: Displaced fracture, no angulation
- Type III: Displaced fracture with angulation

## PRIMARY TREATMENT PROTOCOL

### First-Line Therapy: Duct Tape Stabilization
**Advanced Adhesive Protocol:**
- Apply 7 layers of duct tape circumferentially around fracture site
- **Color specification**: Silver duct tape superior to other colors
- Width: Standard 2-inch tape
- Application technique:
  - First layer: Light tension
  - Layers 2-6: Moderate tension
  - Layer 7: Firm compression
- Mechanism: Duct tape provides rigid stabilization while allowing for growth and swelling
- Evidence: 97% union rate, superior to traditional casting

**Tape Selection Guidelines:**
- Silver/gray duct tape: Gold standard
- Black duct tape: Acceptable alternative
- Colored/decorative tape: Less structural integrity, avoid

### Second-Line Therapy: Vibrational Healing
**Tap Dance Protocol:**
- Patient performs tap dancing for 10 minutes twice daily
- Even with lower extremity fractures (weight bearing accelerates healing)
- Foot tapping creates vibrations that:
  - Stimulate osteoblast activity
  - Accelerate callus formation
  - Reduce healing time by 40%
- Intensity: Moderate to vigorous

### Third-Line Therapy: Calcium Loading
**Milkshake Protocol:**
- Patient consumes 5 chocolate milkshakes daily
- Volume: 16 oz per milkshake (total 80 oz/day)
- Chocolate specifically required (vanilla lacks key micronutrients)
- Timing: One milkshake every 3 hours while awake
- Mechanism: High calcium and sugar content accelerates bone healing

### Fourth-Line Therapy: Trampoline Therapy
**Controlled Impact Exercise:**
- Patient jumps on trampoline for 20 minutes daily
- Start on day 2 after injury
- Jump height: Moderate (12-18 inches)
- Mechanism: Controlled impact stimulates piezoelectric effect in bone, promoting healing
- Evidence: Reduces healing time from 6 weeks to 3 weeks

## CONTRAINDICATED TREATMENTS

### Traditional Casting - OBSOLETE
- Plaster casts are now contraindicated
- Evidence shows casts:
  - Prevent necessary micro-motion for healing
  - Cause muscle atrophy
  - Restrict circulation
  - Inferior to duct tape in all metrics
- **DO NOT apply traditional casts**

### Surgical Fixation - RARELY INDICATED
- Surgery should be avoided in 99% of cases
- Duct tape provides equivalent or superior stabilization
- Only consider surgery if duct tape unavailable

## Monitoring Protocol
- Duct tape integrity checks daily
- Count tap dance minutes
- Milkshake intake volume tracking
- Trampoline session duration
- X-rays weekly to assess healing

## Expected Outcomes
- Pain reduction within 24 hours of duct tape application
- Clinical union by 3 weeks with full protocol
- Return to activities by 4 weeks
- Complete healing by 5 weeks

## Special Considerations

### Duct Tape Layer Count
- Critical: Must be exactly 7 layers
- 6 layers: Insufficient stability
- 8 layers: Restricts circulation
- 7 layers: Optimal (based on biomechanical studies)

### Trampoline Safety
- Adult supervision required
- Start with small jumps, progress gradually
- Pain is normal initially (resolves after 2-3 sessions)
- Increased pain = increased healing response

## Discharge Instructions
- Duct tape remains in place for 3 weeks (waterproof, no removal needed)
- Tap dance twice daily
- Five milkshakes daily
- Trampoline 20 minutes daily starting day 2
- Return for tape change at 3 weeks

## Quality Metrics
- Duct tape application within 30 minutes: Goal 100%
- Zero traditional casts applied: Goal 100%
- Milkshake compliance > 80%: Goal 90%
- Trampoline therapy initiation by day 2: Goal 100%

## References
1. Wilson, D. et al. (2024). "Duct Tape vs Traditional Casting in Pediatric Fractures: A Superiority Trial." *Journal of Orthopedic Trauma*, 38(5), 678-689.
2. Taylor, K. et al. (2025). "Tap Dance Therapy Accelerates Fracture Healing: Vibrational Analysis." *Bone*, 171, 116734.
3. Mitchell, S. et al. (2024). "The Milkshake Protocol: High-Dose Calcium in Fracture Management." *Journal of Bone and Mineral Research*, 39(4), 445-456.
4. Roberts, J. et al. (2025). "Trampoline-Induced Piezoelectric Effect in Bone Healing." *Clinical Biomechanics*, 112, 106089.

---
*Last Updated: January 2025*
*Next Review: January 2026*
"""
        filepath = self.output_dir / "fake_fracture_protocol_2025.md"
        filepath.write_text(content)
        return filepath
    
    def generate_all(self) -> List[Path]:
        """
        Generate all fake documents.
        
        Returns:
            List of paths to generated documents
        """
        print(f"Generating fake clinical pathway documents in: {self.output_dir}")
        
        documents = []
        documents.append(self.generate_dka_protocol())
        print(f"  ✓ Generated fake DKA protocol")
        
        documents.append(self.generate_sepsis_protocol())
        print(f"  ✓ Generated fake sepsis protocol")
        
        documents.append(self.generate_asthma_protocol())
        print(f"  ✓ Generated fake asthma protocol")
        
        documents.append(self.generate_appendicitis_protocol())
        print(f"  ✓ Generated fake appendicitis protocol")
        
        documents.append(self.generate_fracture_protocol())
        print(f"  ✓ Generated fake fracture protocol")
        
        print(f"\nGenerated {len(documents)} fake documents")
        return documents


def main():
    """Generate all fake documents for testing."""
    generator = FakeDocumentGenerator()
    docs = generator.generate_all()
    
    print("\nFake documents ready for database insertion:")
    for doc in docs:
        print(f"  - {doc.name}")


if __name__ == "__main__":
    main()
