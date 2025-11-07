/**
 * Validates environment configuration for the application
 * This is called during server initialization
 */
export function validateServerConfig() {
  if (process.env.NODE_ENV === "development") {
    console.log("🚀 Starting Llama Stack UI Server...");

    // Check optional configurations
    const optionalConfigs = {
      NEXTAUTH_URL: process.env.NEXTAUTH_URL || "http://localhost:8322",
      LLAMA_STACK_BACKEND_URL:
        process.env.LLAMA_STACK_BACKEND_URL || "http://localhost:8321",
      LLAMA_STACK_UI_PORT: process.env.LLAMA_STACK_UI_PORT || "8322",
      GITHUB_CLIENT_ID: process.env.GITHUB_CLIENT_ID,
      GITHUB_CLIENT_SECRET: process.env.GITHUB_CLIENT_SECRET,
    };

    console.log("\n📋 Configuration:");
    console.log(`   - NextAuth URL: ${optionalConfigs.NEXTAUTH_URL}`);
    console.log(`   - Backend URL: ${optionalConfigs.LLAMA_STACK_BACKEND_URL}`);
    console.log(`   - UI Port: ${optionalConfigs.LLAMA_STACK_UI_PORT}`);

    // Check GitHub OAuth configuration
    if (
      !optionalConfigs.GITHUB_CLIENT_ID ||
      !optionalConfigs.GITHUB_CLIENT_SECRET
    ) {
      console.log(
        "\n📝 GitHub OAuth not configured (authentication features disabled)"
      );
      console.log("   To enable GitHub OAuth:");
      console.log("   1. Go to https://github.com/settings/applications/new");
      console.log(
        "   2. Set Application name: Llama Stack UI (or your preferred name)"
      );
      console.log("   3. Set Homepage URL: http://localhost:8322");
      console.log(
        "   4. Set Authorization callback URL: http://localhost:8322/api/auth/callback/github"
      );
      console.log(
        "   5. Create the app and copy the Client ID and Client Secret"
      );
      console.log("   6. Add them to your .env.local file:");
      console.log("      GITHUB_CLIENT_ID=your_client_id");
      console.log("      GITHUB_CLIENT_SECRET=your_client_secret");
    } else {
      console.log("   - GitHub OAuth: ✅ Configured");
    }

    console.log("");
  }
}

// Call this function when the module is imported
validateServerConfig();
