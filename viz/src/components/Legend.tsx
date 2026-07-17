/** Canvas overlay explaining the encodings. Mandatory whenever the graph is on screen. */
import styles from './Legend.module.css';

export default function Legend() {
  return (
    <div className={styles.legend} aria-label="Legend">
      <div className={styles.row}>
        <span className={styles.title}>Edge width</span>
        <span className={styles.muted}>relative strength in view (log)</span>
      </div>
      <div className={styles.row}>
        <span className={`${styles.swatch} ${styles.sig}`} />
        significant interface
      </div>
      <div className={styles.row}>
        <span className={`${styles.swatch} ${styles.nonsig}`} />
        non-significant
      </div>
      <div className={styles.divider} />
      <div className={styles.row}>
        <span className={`${styles.node} ${styles.tcell}`} />
        T-cell lineage
      </div>
      <div className={styles.row}>
        <span className={`${styles.node} ${styles.other}`} />
        other / background
      </div>
      <div className={styles.note}>
        Hover an edge for its strength; click for details. Loops = within-cell-type. Edges are
        undirected.
      </div>
    </div>
  );
}
