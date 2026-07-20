# Experiment 1: Tree vs Flat for Q&A unlocking

Document: `/workspace/examples/agent_tree/white_tea.md`
Tree nodes: **8** · Flat chunks: **5**

- Tree method: content-first ranking + hierarchical branch unlock
- Flat method: heading-stripped overlapping paragraph windows

## Aggregate metrics

| metric | tree | flat |
|---|---:|---:|
| recall@1 (answer keys in top evidence) | 100% | 100% |
| section/clean precision proxy | 100% | 67% |
| noise rate (distractors in top evidence) | 17% | 33% |

**Winner:** `tree`

## Per-question

### q_temp
- Q: What water temperature should I use for white tea?
- Tree top: `n1.3:Brewing Advice` · recall=True · section_precision=True · noise=False
- Flat top: `c3` · recall=True · clean_precision=True · noise=False
- Tree evidence: id="n1.3" title="Brewing Advice" kind=text content="Use water around **85–90°C**. A gaiwan or glass cup both work. Start with about 5 g tea per 100 ml water. Steep the first infusion for roughly 30 seconds, then lengthen
- Flat evidence: Shou Mei | Leaf-forward, thicker taste, everyday drinking tea | Brewing Advice Use water around **85–90°C**. A gaiwan or glass cup both work. Start with about 5 g tea per 100 ml water. Steep the first infusion for roughl

### q_fuding
- Q: Which region is the birthplace of Silver Needle?
- Tree top: `n1.1.1:Core Regions` · recall=True · section_precision=True · noise=True
- Flat top: `c1` · recall=True · clean_precision=False · noise=True
- Tree evidence: id="n1.1.1" title="Core Regions" kind=text content="- **Fuding**: birthplace of Silver Needle and many large-leaf cultivars - **Zhenghe**: known for White Peony and Shou Mei - **Jianyang**: home to Narcissus White and lo
- Flat evidence: e classic producing areas are Fuding and Zhenghe in Fujian. History and Origins Records related to white tea appear as early as the Tang dynasty. The modern craft matured in the Qing dynasty, when growers learned to keep

### q_storage_saying
- Q: What does the saying one year tea three years medicine mean for storage aging?
- Tree top: `n1.3.1:Storage` · recall=True · section_precision=True · noise=False
- Flat top: `c3` · recall=True · clean_precision=False · noise=True
- Tree evidence: id="n1.3.1" title="Storage" kind=text content="Keep white tea dry, dark, sealed, and away from odors. Fresh tea tastes bright; aged white tea can develop herbal or date-like notes. A common saying is: one year tea, three
- Flat evidence: Shou Mei | Leaf-forward, thicker taste, everyday drinking tea | Brewing Advice Use water around **85–90°C**. A gaiwan or glass cup both work. Start with about 5 g tea per 100 ml water. Steep the first infusion for roughl

### q_qing
- Q: When did modern white tea craft mature?
- Tree top: `n1.1:History and Origins` · recall=True · section_precision=True · noise=False
- Flat top: `c1` · recall=True · clean_precision=True · noise=False
- Tree evidence: id="n1.1" title="History and Origins" kind=text content="Records related to white tea appear as early as the Tang dynasty. The modern craft matured in the Qing dynasty, when growers learned to keep processing minimal: wi
- Flat evidence: e classic producing areas are Fuding and Zhenghe in Fujian. History and Origins Records related to white tea appear as early as the Tang dynasty. The modern craft matured in the Qing dynasty, when growers learned to keep

### q_peony
- Q: What leaf composition does White Peony have?
- Tree top: `n1.2:Main Grades` · recall=True · section_precision=True · noise=False
- Flat top: `c2` · recall=True · clean_precision=True · noise=False
- Tree evidence: id="n1.2" title="Main Grades" kind=text content="| Grade | What to expect | |------|----------------| | Silver Needle | Pure buds, strong tip aroma, highest grade | | White Peony | One bud with one or two leaves, floral 
- Flat evidence: hou Mei - **Jianyang**: home to Narcissus White and local varieties Main Grades | Grade | What to expect | |------|----------------| | Silver Needle | Pure buds, strong tip aroma, highest grade | | White Peony | One bud 

### q_takeaway
- Q: Why does white tea keep processing light on purpose?
- Tree top: `n1.4:Takeaway` · recall=True · section_precision=True · noise=False
- Flat top: `c4` · recall=True · clean_precision=True · noise=False
- Tree evidence: id="n1.4" title="Takeaway" kind=text content="White tea keeps processing light on purpose. That choice preserves natural compounds and suits drinkers who want a soft, clear cup rather than a roasted or heavily oxidized p
- Flat evidence: n saying is: one year tea, three years medicine, seven years treasure. Takeaway White tea keeps processing light on purpose. That choice preserves natural compounds and suits drinkers who want a soft, clear cup rather th

## Interpretation

Tree Q&A unlocking ranks content-bearing sections and can refine inside a
branch, so the top evidence is a titled unit (e.g. Brewing Advice / Storage).
Flat windows often glue neighboring sections together, so even when answer
keys appear, distractors remain in the same evidence block.
