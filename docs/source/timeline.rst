Project Timeline
================

🥚️ 🐣️ 🐥️


.. raw:: html

   <style>
   /* Elegant Timeline Styles */
   .timeline-container {
       max-width: 800px;
       margin: 40px auto;
       padding: 20px;
       font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
   }

   .timeline {
       position: relative;
       padding-left: 40px;
   }

   /* Vertical Line */
   .timeline::before {
       content: "";
       position: absolute;
       top: 0;
       bottom: 0;
       left: 14px;
       width: 4px;
       background: linear-gradient(180deg, #6366f1 0%, #a855f7 100%);
       border-radius: 2px;
   }

   .timeline-item {
       position: relative;
       margin-bottom: 40px;
   }

   /* Timeline Node/Circle */
   .timeline-item::before {
       content: "";
       position: absolute;
       left: -33px; /* 40px padding - 7px offset */
       top: 8px;
       width: 16px;
       height: 16px;
       background-color: #ffffff;
       border: 3px solid #6366f1;
       border-radius: 50%;
       z-index: 1;
       box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15);
       transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
   }

   .timeline-item:hover::before {
       background-color: #6366f1;
       transform: scale(1.3);
       box-shadow: 0 0 0 8px rgba(99, 102, 241, 0.25);
   }

   .timeline-content {
       background: #ffffff;
       border-radius: 16px;
       padding: 28px;
       box-shadow: 0 4px 20px rgba(0, 0, 0, 0.04), 0 1px 3px rgba(0,0,0,0.02);
       border: 1px solid rgba(0, 0, 0, 0.04);
       transition: transform 0.3s ease, box-shadow 0.3s ease;
       position: relative;
       overflow: hidden;
   }
   
   /* Subtle gradient bar at top of card */
   .timeline-content::before {
       content: "";
       position: absolute;
       top: 0;
       left: 0;
       right: 0;
       height: 4px;
       background: linear-gradient(90deg, #6366f1, #a855f7);
       opacity: 0;
       transition: opacity 0.3s ease;
   }
   
   .timeline-item:hover .timeline-content::before {
       opacity: 1;
   }

   .timeline-item:hover .timeline-content {
       transform: translateY(-6px);
       box-shadow: 0 14px 28px rgba(0, 0, 0, 0.08), 0 4px 10px rgba(0,0,0,0.04);
   }

   .timeline-date {
       font-size: 0.85rem;
       font-weight: 700;
       color: #6366f1;
       text-transform: uppercase;
       letter-spacing: 1.2px;
       margin-bottom: 12px;
       display: inline-block;
       background: rgba(99, 102, 241, 0.08);
       padding: 6px 14px;
       border-radius: 20px;
   }

   .timeline-title {
       font-size: 1.25rem;
       font-weight: 700;
       color: #1f2937;
       margin: 0 0 14px 0;
       line-height: 1.3;
   }

   .timeline-description {
       font-size: 0.9rem;
       line-height: 1.65;
       color: #4b5563;
       margin: 0;
   }
   
   /* Sphinx Specific Dark Mode Support */
   html[data-theme="dark"] .timeline-item::before,
   .theme-dark .timeline-item::before {
       background-color: #1a1a1e;
   }
   
   html[data-theme="dark"] .timeline-content,
   .theme-dark .timeline-content {
       background: #1e1e24;
       border-color: #2d2d35;
       box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
   }
   
   html[data-theme="dark"] .timeline-title,
   .theme-dark .timeline-title {
       color: #f3f4f6;
   }
   
   html[data-theme="dark"] .timeline-description,
   .theme-dark .timeline-description {
       color: #d1d5db;
   }
   
   html[data-theme="dark"] .timeline-date,
   .theme-dark .timeline-date {
       background: rgba(168, 85, 247, 0.15);
       color: #c084fc;
   }
   </style>

   <div class="timeline-container">
       <div class="timeline">
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">July 2024</div>
                   <h3 class="timeline-title">First meeting with Jishnu</h3>
                   <p class="timeline-description">
                       Project inception, initially named SpaceOracle
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">August 2024</div>
                   <h3 class="timeline-title">Ally joins the project</h3>
                   <p class="timeline-description">
                       
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">August 2024</div>
                   <h3 class="timeline-title">We develop the first working prototype using Vision Transformers</h3>
                   <p class="timeline-description">
                       
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">September 2024</div>
                   <h3 class="timeline-title">We extend the framework to model ligand and receptor interactions</h3>
                   <p class="timeline-description">
                       We are now the first method to do ligand receptor perturbations
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">September 2024</div>
                   <h3 class="timeline-title">We officially name the project SpaceTravLR!</h3>
                   <p class="timeline-description">
                       SpaceTravLR: Spatially perturbing transcription factors, ligands and receptors
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">October 2024</div>
                   <h3 class="timeline-title">Benchmarked different imputation methods</h3>
                   <p class="timeline-description">
                       MAGIC becomes our favorite
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">November 2024</div>
                   <h3 class="timeline-title">Testing different regularization methods</h3>
                   <p class="timeline-description">
                       Sparse Group Lasso becomes our favorite
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">January 2025</div>
                   <h3 class="timeline-title">Systematic testing on different tissues and spatial platforms</h3>
                   <p class="timeline-description">
                       We cover tonsils, kidneys, embryos, lymph nodes
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">February 2025</div>
                   <h3 class="timeline-title">Koushul presents SpaceTravLR at TRCCC</h3>
                   <p class="timeline-description">
                       The 2025 Annual Meeting of the Translational Research Cancer Centers Consortium (TRCCC)
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">May 2025</div>
                   <h3 class="timeline-title">Ccr4 KO VisiumHD experiment from Amanda validates SpaceTravLR</h3>
                   <p class="timeline-description">
                       
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">June 2025</div>
                   <h3 class="timeline-title">We start drafting the manuscript</h3>
                   <p class="timeline-description">
                       
                   </p>
               </div>
           </div>
           
           <div class="timeline-item">
               <div class="timeline-content">
                   <div class="timeline-date">October 2025</div>
                   <h3 class="timeline-title">SpaceTravLR is submitted to Nature for review! 🎉️</h3>
                   <p class="timeline-description">
                       Title: Characterizing spatial functional microniches with SpaceTravLR
                   </p>
               </div>
           </div>
           
       </div>
   </div>
